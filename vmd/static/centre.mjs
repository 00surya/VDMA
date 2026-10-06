import { el } from './ui.mjs';

function field(label, name, value = '', type = 'text') {
  const node = el('label', label, 'field');
  const input = el('input');
  Object.assign(input, {name, value, type, required: true, maxLength: 100});
  if (type === 'tel') {
    input.pattern = '\\+[1-9][0-9]{7,14}';
    input.placeholder = '+919876543210';
  }
  node.append(input);
  return node;
}

export function centreFields(container, details = {}) {
  container.replaceChildren(field('Centre name', 'centre_name', details.name));
  const list = el('div');
  const add = el('button', '+ Add authority', 'button secondary');
  add.type = 'button';
  function authority(contact = {}) {
    if (list.children.length >= 20) return;
    const row = el('div', null, 'contact-row');
    row.append(field('Authority name', 'authority_name', contact.name),
               field('Authority phone (country code required)', 'authority_phone', contact.phone, 'tel'));
    const remove = el('button', 'Remove authority', 'text-button');
    remove.type = 'button';
    remove.addEventListener('click', () => {
      if (list.children.length > 1) row.remove();
    });
    row.append(remove);
    list.append(row);
  }
  (details.authorities?.length ? details.authorities : [{}]).forEach(authority);
  add.addEventListener('click', () => authority());
  container.append(el('h3', 'Authorities'), list, add, el('h3', 'Hospital'),
    field('Hospital name', 'hospital_name', details.hospital?.name),
    field('Hospital phone (country code required)', 'hospital_phone', details.hospital?.phone, 'tel'));
}

export function readCentre(form) {
  const data = new FormData(form);
  const phones = data.getAll('authority_phone');
  return {name: data.get('centre_name'),
    authorities: data.getAll('authority_name').map((name, i) => ({name, phone: phones[i]})),
    hospital: {name: data.get('hospital_name'), phone: data.get('hospital_phone')}};
}
