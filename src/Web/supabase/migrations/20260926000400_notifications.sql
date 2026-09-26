-- In-app notifications. First kind: "a task was assigned to you". Rows are written only by a trigger on
-- public.task (never by clients); each user reads, marks as read and deletes only their own notifications.
-- The table is in the supabase_realtime publication, so the bell updates live (Realtime applies the RLS below).

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;

create table public.notification (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  kind text not null check (kind in ('task_assigned')),
  task_id bigint references public.task (id) on delete cascade,
  -- display data frozen at creation (task title, target, who assigned it); no authorization is based on it
  payload jsonb not null default '{}'::jsonb,
  read_at timestamptz,
  created_at timestamptz not null default now()
);
create index notification_user_created_idx on public.notification (user_id, created_at desc);
create index notification_user_unread_idx on public.notification (user_id) where read_at is null;
create index notification_task_idx on public.notification (task_id);

alter table public.notification enable row level security;

create policy notification_select on public.notification for select to authenticated
  using (user_id = (select auth.uid()));
create policy notification_update on public.notification for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy notification_delete on public.notification for delete to authenticated
  using (user_id = (select auth.uid()));

revoke all on public.notification from anon, authenticated, public;
grant select, delete on public.notification to authenticated;
-- the only column a user may change: marking as read / unread
grant update (read_at) on public.notification to authenticated;

-- security definer: writes a row for another user (the assignee), which no client may do. It lives in the
-- unexposed schema "private", is not executable by API roles, and only reads the task row it fires on.
create or replace function private.tg_task_notify_assignee()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  actor uuid := (select auth.uid());
  claims jsonb := (select auth.jwt());
begin
  if tg_op = 'UPDATE' and new.assignee is not distinct from old.assignee then
    return null;
  end if;
  -- reassigned: the previous assignee's unread "assigned to you" is no longer true
  if tg_op = 'UPDATE' and old.assignee is not null then
    delete from public.notification n
    where n.user_id = old.assignee and n.task_id = new.id and n.kind = 'task_assigned' and n.read_at is null;
  end if;
  if new.assignee is null then
    return null;
  end if;
  -- assigning a task to yourself needs no notification
  if actor is not null and new.assignee = actor then
    return null;
  end if;

  insert into public.notification (user_id, kind, task_id, payload)
  values (
    new.assignee,
    'task_assigned',
    new.id,
    jsonb_build_object(
      'title', new.title,
      'task_kind', new.kind,
      'target_id', new.target_id,
      'row_id', new.row_id,
      'vineyard_id', new.vineyard_id,
      'gap_length_m', new.gap_length_m,
      'priority', new.priority,
      'due_date', new.due_date,
      'uat_key', new.uat_key,
      'actor_email', claims ->> 'email',
      'actor_name', claims -> 'user_metadata' ->> 'full_name'
    )
  );
  return null;
end
$$;

revoke execute on function private.tg_task_notify_assignee() from public, anon, authenticated;

create trigger task_notify_assignee after insert or update of assignee on public.task
  for each row execute function private.tg_task_notify_assignee();

alter publication supabase_realtime add table public.notification;
