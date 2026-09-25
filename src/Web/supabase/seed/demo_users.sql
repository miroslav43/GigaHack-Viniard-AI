-- Demo accounts for the Solemtrix web app (Supabase Auth, email + password).
-- Authorization lives in raw_app_meta_data (not user-editable): uat = sireti | cojusna, uat_role.
-- Idempotent: re-running updates metadata and password, never duplicates users.
--
-- The password is NOT stored in git: replace __DEMO_PASSWORD__ before running
-- (Supabase SQL editor, MCP execute_sql, or psql with the service connection).

do $$
declare
  demo_password constant text := '__DEMO_PASSWORD__';
  acc record;
  uid uuid;
begin
  if demo_password = '__DEMO_PASSWORD__' then
    raise exception 'set the demo password first';
  end if;

  for acc in
    select * from (values
      ('admin@solemtrix.demo',    'Administrator Solemtrix', 'sireti',  'platform_admin'),
      ('primar@sireti.demo',      'Primar Sireți',           'sireti',  'uat_admin'),
      ('inspector@sireti.demo',   'Inspector Sireți',        'sireti',  'inspector'),
      ('primar@cojusna.demo',     'Primar Cojușna',          'cojusna', 'uat_admin')
    ) as a(email, full_name, uat, uat_role)
  loop
    select id into uid from auth.users where email = acc.email;

    if uid is null then
      uid := gen_random_uuid();
      insert into auth.users (
        instance_id, id, aud, role, email, encrypted_password, email_confirmed_at,
        raw_app_meta_data, raw_user_meta_data, created_at, updated_at,
        confirmation_token, email_change, email_change_token_new, recovery_token
      ) values (
        '00000000-0000-0000-0000-000000000000', uid, 'authenticated', 'authenticated', acc.email,
        extensions.crypt(demo_password, extensions.gen_salt('bf')), now(),
        jsonb_build_object('provider', 'email', 'providers', jsonb_build_array('email'), 'uat', acc.uat, 'uat_role', acc.uat_role),
        jsonb_build_object('full_name', acc.full_name),
        now(), now(), '', '', '', ''
      );
      insert into auth.identities (id, user_id, provider_id, identity_data, provider, last_sign_in_at, created_at, updated_at)
      values (
        gen_random_uuid(), uid, uid::text,
        jsonb_build_object('sub', uid::text, 'email', acc.email, 'email_verified', true),
        'email', now(), now(), now()
      );
    else
      update auth.users set
        encrypted_password = extensions.crypt(demo_password, extensions.gen_salt('bf')),
        email_confirmed_at = coalesce(email_confirmed_at, now()),
        raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
          || jsonb_build_object('uat', acc.uat, 'uat_role', acc.uat_role),
        raw_user_meta_data = coalesce(raw_user_meta_data, '{}'::jsonb) || jsonb_build_object('full_name', acc.full_name),
        updated_at = now()
      where id = uid;
    end if;
  end loop;
end $$;

select email, raw_app_meta_data ->> 'uat' as uat, raw_app_meta_data ->> 'uat_role' as uat_role, email_confirmed_at is not null as confirmed
from auth.users where email like '%.demo' order by email;
