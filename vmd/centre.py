"""One locally registered centre with password sessions and responder contacts."""
import hashlib
import hmac
import json
import secrets
import time
import threading

from pydantic import BaseModel, Field, StrictBool, model_validator


class Contact(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    phone: str = Field(pattern=r'^\+[1-9][0-9]{7,14}$')

    @model_validator(mode='after')
    def clean_name(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError('Contact name is required')
        return self


class CentreDetails(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    authorities: list[Contact] = Field(min_length=1, max_length=20)
    hospital: Contact
    messaging_enabled: StrictBool = False
    calling_enabled: StrictBool = False

    @model_validator(mode='after')
    def clean(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError('Centre name is required')
        phones = [item.phone for item in self.authorities]
        if len(phones) != len(set(phones)):
            raise ValueError('Authority phone numbers must be unique')
        return self


class CentreSignup(CentreDetails):
    password: str = Field(min_length=10, max_length=200)


class CentreLogin(BaseModel):
    password: str = Field(min_length=1, max_length=200)


class Centre:
    def __init__(self, store):
        self.store = store
        self.lock = threading.Lock()
        self.failures = []
        with store.connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS centre (
                    id INTEGER PRIMARY KEY CHECK(id=1), details TEXT NOT NULL,
                    salt TEXT NOT NULL, password_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS centre_sessions (
                    token_hash TEXT PRIMARY KEY, expires REAL NOT NULL);
            ''')

    @staticmethod
    def digest(password, salt):
        return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()

    def details(self):
        with self.store.connect() as conn:
            row = conn.execute('SELECT details FROM centre WHERE id=1').fetchone()
        return json.loads(row[0]) if row else None

    def signup(self, body):
        salt = secrets.token_hex(16)
        hashed = self.digest(body.password, salt)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('SELECT 1 FROM centre').fetchone():
                raise ValueError('This installation already has a centre. Sign in to manage it.')
            conn.execute('INSERT INTO centre VALUES (1,?,?,?)',
                         (json.dumps(body.model_dump(exclude={'password'})), salt, hashed))
        return self.session()

    def login(self, password):
        with self.lock:
            now = time.time()
            self.failures = [stamp for stamp in self.failures if now-stamp < 60]
            if len(self.failures) >= 5:
                raise ValueError('Too many attempts. Wait one minute before trying again.')
            with self.store.connect() as conn:
                row = conn.execute('SELECT salt,password_hash FROM centre WHERE id=1').fetchone()
            if not row or not hmac.compare_digest(self.digest(password, row[0]), row[1]):
                self.failures.append(now)
                raise ValueError('Incorrect password or centre not registered')
            self.failures.clear()
        return self.session()

    def session(self):
        token = secrets.token_urlsafe(32)
        with self.store.connect() as conn:
            conn.execute('DELETE FROM centre_sessions WHERE expires<=?', (time.time(),))
            conn.execute('INSERT INTO centre_sessions VALUES (?,?)',
                         (hashlib.sha256(token.encode()).hexdigest(), time.time()+86400))
        return token

    def authenticated(self, token):
        if not token or len(token) > 100:
            return False
        with self.store.connect() as conn:
            return bool(conn.execute('SELECT 1 FROM centre_sessions WHERE token_hash=? AND expires>?',
                        (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone())

    def logout(self, token):
        with self.store.connect() as conn:
            conn.execute('DELETE FROM centre_sessions WHERE token_hash=?',
                         (hashlib.sha256((token or '').encode()).hexdigest(),))

    def update(self, body):
        with self.store.connect() as conn:
            conn.execute('UPDATE centre SET details=? WHERE id=1', (body.model_dump_json(),))
        return self.details()
