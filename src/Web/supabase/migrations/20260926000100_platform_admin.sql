-- Platform administration: municipalities (UAT), surveys, admin audit log.
-- Authorization: JWT app_metadata (server-set, not user-editable): uat = <uat key>, uat_role = platform_admin | uat_admin | inspector | viewer.
-- Geometries in EPSG:4326 (UATs may lie in UTM 34N or 35N); areas computed on the geography (m²).

create extension if not exists postgis with schema extensions;

-- ---------------------------------------------------------------------------
-- helpers (SECURITY INVOKER; read only the caller's own JWT)
-- ---------------------------------------------------------------------------
create or replace function public.is_platform_admin()
returns boolean
language sql
stable
set search_path = ''
as $$
  select coalesce((select auth.jwt()) -> 'app_metadata' ->> 'uat_role', '') = 'platform_admin'
$$;

create or replace function public.my_uat_key()
returns text
language sql
stable
set search_path = ''
as $$
  select (select auth.jwt()) -> 'app_metadata' ->> 'uat'
$$;

-- keep geometries valid multipolygons and derive the area (one function per table: PL/pgSQL
-- cannot reference a column the row type does not have, even in an untaken branch)
create or replace function public.tg_uat_area()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.geofence := extensions.st_multi(extensions.st_collectionextract(extensions.st_makevalid(new.geofence), 3));
  new.area_ha := extensions.st_area(new.geofence::extensions.geography) / 10000.0;
  return new;
end
$$;

create or replace function public.tg_survey_area()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.footprint := extensions.st_multi(extensions.st_collectionextract(extensions.st_makevalid(new.footprint), 3));
  new.area_ha := extensions.st_area(new.footprint::extensions.geography) / 10000.0;
  return new;
end
$$;

-- ---------------------------------------------------------------------------
-- tables
-- ---------------------------------------------------------------------------
create table public.uat (
  key text primary key check (key ~ '^[a-z0-9][a-z0-9-]{1,39}$'),
  name text not null check (char_length(name) between 2 and 120),
  district text check (char_length(district) <= 120),
  country text not null default 'MD' check (country in ('MD', 'RO')),
  osm_relation_id bigint unique,
  geofence extensions.geometry(MultiPolygon, 4326) not null,
  area_ha double precision not null default 0,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  created_by uuid references auth.users (id) on delete set null
);
create index uat_geofence_gix on public.uat using gist (geofence);
create trigger uat_area before insert or update of geofence on public.uat
  for each row execute function public.tg_uat_area();

create table public.survey (
  id text primary key check (id ~ '^[a-z0-9][a-z0-9-]{1,39}$'),
  name text not null check (char_length(name) between 2 and 160),
  captured_at date,
  source text,
  license text,
  data_path text not null check (data_path ~ '^/data/[a-z0-9-]+$'),
  footprint extensions.geometry(MultiPolygon, 4326) not null,
  area_ha double precision not null default 0,
  created_at timestamptz not null default now()
);
create index survey_footprint_gix on public.survey using gist (footprint);
create trigger survey_area before insert or update of footprint on public.survey
  for each row execute function public.tg_survey_area();

create table public.admin_audit_log (
  id bigint generated always as identity primary key,
  at timestamptz not null default now(),
  actor uuid default auth.uid() references auth.users (id) on delete set null,
  actor_email text default (auth.jwt() ->> 'email'),
  action text not null check (char_length(action) <= 64),
  entity text not null check (char_length(entity) <= 64),
  entity_id text,
  details jsonb not null default '{}'::jsonb
);
create index admin_audit_log_at_idx on public.admin_audit_log (at desc);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
alter table public.uat enable row level security;
alter table public.survey enable row level security;
alter table public.admin_audit_log enable row level security;

-- a user reads their own municipality; the platform admin reads and writes all
create policy uat_select on public.uat for select to authenticated
  using ((select public.is_platform_admin()) or key = (select public.my_uat_key()));
create policy uat_insert on public.uat for insert to authenticated
  with check ((select public.is_platform_admin()));
create policy uat_update on public.uat for update to authenticated
  using ((select public.is_platform_admin())) with check ((select public.is_platform_admin()));
create policy uat_delete on public.uat for delete to authenticated
  using ((select public.is_platform_admin()));

-- a survey is visible to a municipality only if it intersects its boundary
create policy survey_select on public.survey for select to authenticated
  using (
    (select public.is_platform_admin())
    or exists (
      select 1 from public.uat u
      where u.key = (select public.my_uat_key()) and u.active and extensions.st_intersects(u.geofence, survey.footprint)
    )
  );
create policy survey_insert on public.survey for insert to authenticated
  with check ((select public.is_platform_admin()));
create policy survey_update on public.survey for update to authenticated
  using ((select public.is_platform_admin())) with check ((select public.is_platform_admin()));
create policy survey_delete on public.survey for delete to authenticated
  using ((select public.is_platform_admin()));

-- append-only audit log, platform admin only
create policy audit_select on public.admin_audit_log for select to authenticated
  using ((select public.is_platform_admin()));
create policy audit_insert on public.admin_audit_log for insert to authenticated
  with check ((select public.is_platform_admin()) and actor = (select auth.uid()));

-- ---------------------------------------------------------------------------
-- read views (security_invoker: RLS of the caller applies) with GeoJSON output
-- ---------------------------------------------------------------------------
create view public.uat_public with (security_invoker = true) as
  select key, name, district, country, osm_relation_id, round(area_ha::numeric, 2) as area_ha, active, created_at,
         extensions.st_asgeojson(geofence, 7)::jsonb as geofence
  from public.uat;

create view public.survey_public with (security_invoker = true) as
  select id, name, captured_at, source, license, data_path, round(area_ha::numeric, 2) as area_ha, created_at,
         extensions.st_asgeojson(footprint, 7)::jsonb as footprint
  from public.survey;

create view public.uat_survey with (security_invoker = true) as
  select u.key as uat_key, s.id as survey_id, s.name as survey_name, s.data_path,
         round((extensions.st_area(extensions.st_intersection(u.geofence, s.footprint)::extensions.geography) / 10000.0)::numeric, 2) as overlap_ha
  from public.uat u
  join public.survey s on extensions.st_intersects(u.geofence, s.footprint)
  where u.active;

-- ---------------------------------------------------------------------------
-- write RPCs (SECURITY INVOKER: RLS decides; geometry comes in as GeoJSON, Polygon or MultiPolygon —
-- st_multi before the column's MultiPolygon type check, which runs before the BEFORE trigger)
-- ---------------------------------------------------------------------------
create or replace function public.enroll_uat(
  p_key text, p_name text, p_district text, p_country text, p_osm_relation_id bigint, p_geofence jsonb
) returns text
language plpgsql
set search_path = ''
as $$
begin
  insert into public.uat (key, name, district, country, osm_relation_id, geofence, created_by)
  values (p_key, p_name, p_district, p_country, p_osm_relation_id,
          extensions.st_multi(extensions.st_setsrid(extensions.st_geomfromgeojson(p_geofence::text), 4326)), (select auth.uid()));
  return p_key;
end
$$;

create or replace function public.upsert_survey(
  p_id text, p_name text, p_captured_at date, p_source text, p_license text, p_data_path text, p_footprint jsonb
) returns text
language plpgsql
set search_path = ''
as $$
begin
  insert into public.survey (id, name, captured_at, source, license, data_path, footprint)
  values (p_id, p_name, p_captured_at, p_source, p_license, p_data_path,
          extensions.st_multi(extensions.st_setsrid(extensions.st_geomfromgeojson(p_footprint::text), 4326)))
  on conflict (id) do update set
    name = excluded.name, captured_at = excluded.captured_at, source = excluded.source,
    license = excluded.license, data_path = excluded.data_path, footprint = excluded.footprint;
  return p_id;
end
$$;

-- ---------------------------------------------------------------------------
-- privileges: nothing for anon; authenticated goes through RLS
-- ---------------------------------------------------------------------------
revoke all on public.uat, public.survey, public.admin_audit_log from anon, public;
revoke all on public.uat_public, public.survey_public, public.uat_survey from anon, public;
grant select, insert, update, delete on public.uat, public.survey to authenticated;
grant select, insert on public.admin_audit_log to authenticated;
grant select on public.uat_public, public.survey_public, public.uat_survey to authenticated;

revoke execute on function public.is_platform_admin(), public.my_uat_key(),
  public.enroll_uat(text, text, text, text, bigint, jsonb),
  public.upsert_survey(text, text, date, text, text, text, jsonb) from anon, public;
grant execute on function public.is_platform_admin(), public.my_uat_key(),
  public.enroll_uat(text, text, text, text, bigint, jsonb),
  public.upsert_survey(text, text, date, text, text, text, jsonb) to authenticated;
revoke execute on function public.tg_uat_area(), public.tg_survey_area() from anon, authenticated, public;
