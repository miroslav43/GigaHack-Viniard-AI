-- Requests from the presentation site ("request a demo / a pilot"). Written only by the server action of the
-- public page with the secret key (no insert grant to anon/authenticated: the public REST API cannot spam it);
-- read and handled only by platform admins.

create table public.lead (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  name text not null check (char_length(name) between 2 and 120),
  institution text not null check (char_length(institution) between 2 and 200),
  institution_type text not null check (institution_type in ('primarie', 'consiliu_raional', 'minister_agentie', 'asociatie_producatori', 'altul')),
  position text check (char_length(position) <= 120),
  email text not null check (char_length(email) between 5 and 200 and email ~ '^[^@\s]+@[^@\s]+\.[^@\s]+$'),
  phone text check (char_length(phone) <= 40),
  message text check (char_length(message) <= 2000),
  locale text not null default 'ro' check (locale in ('ro', 'en', 'ru')),
  consent boolean not null check (consent),   -- agreed to be contacted about this request
  status text not null default 'new' check (status in ('new', 'contacted', 'pilot', 'closed')),
  notes text check (char_length(notes) <= 2000)
);
create index lead_created_idx on public.lead (created_at desc);

alter table public.lead enable row level security;

create policy lead_select on public.lead for select to authenticated using ((select public.is_platform_admin()));
create policy lead_update on public.lead for update to authenticated
  using ((select public.is_platform_admin())) with check ((select public.is_platform_admin()));

revoke all on public.lead from anon, authenticated, public;
grant select on public.lead to authenticated;
grant update (status, notes) on public.lead to authenticated;
