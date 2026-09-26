-- Fix: with search_path = '' the row comparison including the geometry column had no resolvable `=` operator
-- ("operator is not unique: extensions.geometry = extensions.geometry"), so every assignee update failed.
-- Compare the geometry through its EWKB bytes instead.
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
        new.gap_length_m, new.assignee, new.assignee_email, new.priority, new.due_date, new.created_by, new.created_at)
       is distinct from
       (old.uat_key, old.survey_id, old.target_id, old.kind, old.title, old.description, old.vineyard_id, old.row_id,
        old.gap_length_m, old.assignee, old.assignee_email, old.priority, old.due_date, old.created_by, old.created_at)
       or extensions.st_asewkb(new.location) is distinct from extensions.st_asewkb(old.location) then
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

revoke execute on function public.tg_task_guard() from anon, authenticated, public;
