-- Field tasks: inspection targets from the AI analysis (row gaps, missing plants, waste) become tasks that the
-- municipality (UAT) admin assigns to the field team. Authorization from JWT app_metadata (uat, uat_role).

create or replace function public.my_uat_role()
returns text
language sql
stable
set search_path = ''
as $$
  select (select auth.jwt()) -> 'app_metadata' ->> 'uat_role'
$$;

create type public.task_status as enum ('open', 'in_progress', 'done', 'cancelled');
create type public.task_kind as enum ('gap', 'missing', 'waste', 'other');

create table public.task (
  id bigint generated always as identity primary key,
  uat_key text not null references public.uat (key) on delete cascade,
  survey_id text references public.survey (id) on delete set null,
  target_id text,                                   -- id in the survey bundle, e.g. T003
  kind public.task_kind not null,
  title text not null check (char_length(title) between 2 and 200),
  description text check (char_length(description) <= 2000),
  vineyard_id text,
  row_id text,
  gap_length_m double precision,
  location extensions.geometry(Point, 4326),
  assignee uuid references auth.users (id) on delete set null,
  assignee_email text,                              -- display only (auth.users is not readable by clients)
  status public.task_status not null default 'open',
  priority smallint not null default 2 check (priority between 1 and 3),
  due_date date,
  resolution_note text check (char_length(resolution_note) <= 2000),
  created_by uuid default auth.uid() references auth.users (id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz,
  unique (uat_key, survey_id, target_id)
);
create index task_uat_status_idx on public.task (uat_key, status);
create index task_assignee_idx on public.task (assignee);
create index task_location_gix on public.task using gist (location);

-- keeps a task inside its municipality, maintains timestamps, and limits what a non-admin assignee may change
create or replace function public.tg_task_guard()
returns trigger
language plpgsql
set search_path = ''
as $$
declare
  is_admin boolean := (select public.is_platform_admin()) or (select public.my_uat_role()) = 'uat_admin';
begin
  if new.location is not null and not exists (
    select 1 from public.uat u where u.key = new.uat_key and extensions.st_intersects(u.geofence, new.location)
  ) then
    raise exception 'task location is outside the municipality boundary' using errcode = '23514';
  end if;

  if tg_op = 'UPDATE' and not is_admin then
    -- the assignee may only move the task forward and write a note
    if (new.uat_key, new.survey_id, new.target_id, new.kind, new.title, new.description, new.vineyard_id, new.row_id,
        new.location, new.assignee, new.assignee_email, new.priority, new.due_date, new.created_by, new.created_at)
       is distinct from
       (old.uat_key, old.survey_id, old.target_id, old.kind, old.title, old.description, old.vineyard_id, old.row_id,
        old.location, old.assignee, old.assignee_email, old.priority, old.due_date, old.created_by, old.created_at) then
      raise exception 'only status and resolution note can be changed by the assignee' using errcode = '42501';
    end if;
    if new.status = 'cancelled' then
      raise exception 'only the municipality admin can cancel a task' using errcode = '42501';
    end if;
  end if;

  new.updated_at := now();
  if new.status = 'done' and (tg_op = 'INSERT' or old.status is distinct from 'done') then
    new.completed_at := now();
  elsif new.status <> 'done' then
    new.completed_at := null;
  end if;
  return new;
end
$$;

create trigger task_guard before insert or update on public.task
  for each row execute function public.tg_task_guard();

alter table public.task enable row level security;

-- everyone in the municipality reads its tasks; the platform admin reads all
create policy task_select on public.task for select to authenticated
  using ((select public.is_platform_admin()) or uat_key = (select public.my_uat_key()));
-- only the municipality admin creates and deletes tasks, and only in its own municipality
create policy task_insert on public.task for insert to authenticated
  with check (
    (select public.is_platform_admin())
    or (uat_key = (select public.my_uat_key()) and (select public.my_uat_role()) = 'uat_admin')
  );
create policy task_delete on public.task for delete to authenticated
  using (
    (select public.is_platform_admin())
    or (uat_key = (select public.my_uat_key()) and (select public.my_uat_role()) = 'uat_admin')
  );
-- the municipality admin updates any task of its municipality; an inspector only the tasks assigned to them
-- (the trigger limits the inspector to status + note)
create policy task_update on public.task for update to authenticated
  using (
    (select public.is_platform_admin())
    or (uat_key = (select public.my_uat_key()) and (
      (select public.my_uat_role()) = 'uat_admin'
      or (assignee = (select auth.uid()) and (select public.my_uat_role()) = 'inspector')
    ))
  )
  with check (
    (select public.is_platform_admin())
    or (uat_key = (select public.my_uat_key()) and (
      (select public.my_uat_role()) = 'uat_admin'
      or (assignee = (select auth.uid()) and (select public.my_uat_role()) = 'inspector')
    ))
  );

create view public.task_public with (security_invoker = true) as
  select id, uat_key, survey_id, target_id, kind, title, description, vineyard_id, row_id, gap_length_m,
         extensions.st_x(location) as lon, extensions.st_y(location) as lat,
         assignee, assignee_email, status, priority, due_date, resolution_note,
         created_by, created_at, updated_at, completed_at
  from public.task;

revoke all on public.task, public.task_public from anon, public;
grant select, insert, update, delete on public.task to authenticated;
grant select on public.task_public to authenticated;
revoke execute on function public.my_uat_role() from anon, public;
grant execute on function public.my_uat_role() to authenticated;
revoke execute on function public.tg_task_guard() from anon, authenticated, public;
