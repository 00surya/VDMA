"""Authenticated test centre; contacts are fictional and messaging stays disabled."""
TEST_CENTRE = {'name': 'Test centre', 'password': 'test-password-12345',
               'authorities': [{'name': 'Test authority', 'phone': '+12025550101'}],
               'hospital': {'name': 'Test hospital', 'phone': '+12025550102'}}


def authenticate(client):
    path = '/api/auth/login' if client.get('/api/auth/status').json()['registered'] else '/api/auth/signup'
    body = {'password': TEST_CENTRE['password']} if path.endswith('login') else TEST_CENTRE
    result = client.post(path, json=body, headers={'x-vmd-client': 'dashboard'})
    assert result.status_code == 200, result.text
