"use server";

// Platform-admin server actions. Every action re-checks the caller on the server:
// UAT/survey writes go through the caller's session (RLS: platform_admin only); user management uses the
// Admin API (secret key) after re-reading the caller's app_metadata from Auth (not trusting a stale JWT).
import { revalidatePath } from "next/cache";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { Geometry } from "geojson";
import { createClient } from "@/lib/supabase/server";
import { ADMIN_API_ENABLED, createAdminClient } from "@/lib/supabase/admin";
import { getViewer, isPlatformAdmin, type UatRole } from "@/lib/viewer";
import { fetchOsmBoundary, searchOsmBoundaries, type OsmBoundary, type OsmSearchHit } from "@/lib/osm";

export type ActionResult<T = null> = { ok: true; data: T } | { ok: false; error: string };

const ROLES: UatRole[] = ["platform_admin", "uat_admin", "inspector", "viewer"];
const KEY_RE = /^[a-z0-9][a-z0-9-]{1,39}$/;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const ok = <T,>(data: T): ActionResult<T> => ({ ok: true, data });
const fail = (error: string): ActionResult<never> => ({ ok: false, error });
const done = () => revalidatePath("/[locale]/super-admin", "page");

async function requireAdmin() {
  const viewer = await getViewer();
  if (!isPlatformAdmin(viewer)) throw new Error("forbidden");
  return { viewer, supabase: await createClient() };
}

/** For Admin API actions: confirm the role from Auth itself, not only from the (up to 1 h old) JWT. */
async function requireAdminStrict() {
  const ctx = await requireAdmin();
  if (!ADMIN_API_ENABLED) throw new Error("secret_key_missing");
  const admin = createAdminClient();
  const { data, error } = await admin.auth.admin.getUserById(ctx.viewer.userId!);
  if (error || data.user?.app_metadata?.uat_role !== "platform_admin") throw new Error("forbidden");
  return { ...ctx, admin };
}

async function audit(action: string, entity: string, entityId: string | null, details: Record<string, unknown> = {}) {
  const supabase = await createClient();
  const { viewer } = await requireAdmin();
  await supabase.from("admin_audit_log").insert({ action, entity, entity_id: entityId, details, actor: viewer.userId });
}

const run = async <T,>(fn: () => Promise<T>): Promise<ActionResult<T>> => {
  try {
    return ok(await fn());
  } catch (e) {
    return fail(e instanceof Error ? e.message : String(e));
  }
};

// ---------------------------------------------------------------- UAT ----

export async function searchOsm(query: string): Promise<ActionResult<OsmSearchHit[]>> {
  return run(async () => {
    await requireAdmin();
    if (query.trim().length < 2) return [];
    return searchOsmBoundaries(query.trim());
  });
}

export async function previewOsm(osmRelationId: number): Promise<ActionResult<OsmBoundary>> {
  return run(async () => {
    await requireAdmin();
    return fetchOsmBoundary(osmRelationId);
  });
}

export async function enrollUat(input: {
  key: string;
  name: string;
  district: string;
  country: "MD" | "RO";
  osmRelationId: number;
}): Promise<ActionResult<string>> {
  return run(async () => {
    const { supabase } = await requireAdmin();
    const key = input.key.trim().toLowerCase();
    if (!KEY_RE.test(key)) throw new Error("invalid_key");
    if (input.name.trim().length < 2) throw new Error("invalid_name");
    if (!["MD", "RO"].includes(input.country)) throw new Error("invalid_country");
    // the boundary is always re-fetched on the server — the client never supplies geometry
    const boundary = await fetchOsmBoundary(input.osmRelationId);
    const { error } = await supabase.rpc("enroll_uat", {
      p_key: key,
      p_name: input.name.trim(),
      p_district: input.district.trim() || null,
      p_country: input.country,
      p_osm_relation_id: input.osmRelationId,
      p_geofence: boundary.geometry,
    });
    if (error) throw new Error(error.code === "23505" ? "duplicate" : error.message);
    await audit("uat.enroll", "uat", key, { name: input.name, osm_relation_id: input.osmRelationId, country: input.country });
    done();
    return key;
  });
}

export async function setUatActive(key: string, active: boolean): Promise<ActionResult> {
  return run(async () => {
    const { supabase } = await requireAdmin();
    const { error } = await supabase.from("uat").update({ active }).eq("key", key);
    if (error) throw new Error(error.message);
    await audit(active ? "uat.activate" : "uat.deactivate", "uat", key);
    done();
    return null;
  });
}

export async function deleteUat(key: string): Promise<ActionResult> {
  return run(async () => {
    const { supabase } = await requireAdmin();
    if (ADMIN_API_ENABLED) {
      const { data } = await createAdminClient().auth.admin.listUsers({ perPage: 1000 });
      if (data.users.some((u) => u.app_metadata?.uat === key)) throw new Error("uat_has_users");
    }
    const { error } = await supabase.from("uat").delete().eq("key", key);
    if (error) throw new Error(error.message);
    await audit("uat.delete", "uat", key);
    done();
    return null;
  });
}

// ------------------------------------------------------------- surveys ----

/** Registers (or refreshes) a static survey bundle from public/data/<id>; footprint = the organizer study area. */
export async function registerSurvey(surveyId: string): Promise<ActionResult<string>> {
  return run(async () => {
    const { supabase } = await requireAdmin();
    if (!KEY_RE.test(surveyId)) throw new Error("invalid_key");
    const root = path.join(process.cwd(), "public", "data");
    const summary = JSON.parse(await readFile(path.join(root, surveyId, "summary.json"), "utf8"));
    const study = JSON.parse(await readFile(path.join(root, "ref", "study_area.geojson"), "utf8"));
    const { error } = await supabase.rpc("upsert_survey", {
      p_id: surveyId,
      p_name: summary.survey.name,
      p_captured_at: summary.survey.captured_at,
      p_source: summary.survey.source,
      p_license: summary.survey.license,
      p_data_path: `/data/${surveyId}`,
      p_footprint: study.features[0].geometry as Geometry,
    });
    if (error) throw new Error(error.message);
    await audit("survey.register", "survey", surveyId, { name: summary.survey.name });
    done();
    return surveyId;
  });
}

export async function deleteSurvey(surveyId: string): Promise<ActionResult> {
  return run(async () => {
    const { supabase } = await requireAdmin();
    const { error } = await supabase.from("survey").delete().eq("id", surveyId);
    if (error) throw new Error(error.message);
    await audit("survey.delete", "survey", surveyId);
    done();
    return null;
  });
}

// --------------------------------------------------------------- users ----

function checkAccess(uat: string | null, role: string) {
  if (!ROLES.includes(role as UatRole)) throw new Error("invalid_role");
  if (uat !== null && !KEY_RE.test(uat)) throw new Error("invalid_key");
}

export async function createUser(input: {
  email: string;
  password: string;
  fullName: string;
  uat: string | null;
  role: UatRole;
}): Promise<ActionResult<string>> {
  return run(async () => {
    const { admin } = await requireAdminStrict();
    const email = input.email.trim().toLowerCase();
    if (!EMAIL_RE.test(email)) throw new Error("invalid_email");
    if (input.password.length < 10) throw new Error("weak_password");
    checkAccess(input.uat, input.role);
    const { data, error } = await admin.auth.admin.createUser({
      email,
      password: input.password,
      email_confirm: true,
      user_metadata: { full_name: input.fullName.trim() || undefined },
      app_metadata: { uat: input.uat, uat_role: input.role },
    });
    if (error) throw new Error(error.code === "email_exists" ? "email_exists" : error.message);
    await audit("user.create", "user", data.user.id, { email, uat: input.uat, role: input.role });
    done();
    return data.user.id;
  });
}

export async function updateUserAccess(input: {
  id: string;
  fullName: string;
  uat: string | null;
  role: UatRole;
}): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer } = await requireAdminStrict();
    checkAccess(input.uat, input.role);
    if (input.id === viewer.userId && input.role !== "platform_admin") throw new Error("self_demotion");
    const { error } = await admin.auth.admin.updateUserById(input.id, {
      user_metadata: { full_name: input.fullName.trim() || null },
      app_metadata: { uat: input.uat, uat_role: input.role },
    });
    if (error) throw new Error(error.message);
    await audit("user.update_access", "user", input.id, { uat: input.uat, role: input.role });
    done();
    return null;
  });
}

export async function resetUserPassword(id: string, password: string): Promise<ActionResult> {
  return run(async () => {
    const { admin } = await requireAdminStrict();
    if (password.length < 10) throw new Error("weak_password");
    const { error } = await admin.auth.admin.updateUserById(id, { password });
    if (error) throw new Error(error.message);
    await audit("user.reset_password", "user", id);
    done();
    return null;
  });
}

export async function setUserBanned(id: string, banned: boolean): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer } = await requireAdminStrict();
    if (id === viewer.userId) throw new Error("self_ban");
    const { error } = await admin.auth.admin.updateUserById(id, { ban_duration: banned ? "876000h" : "none" });
    if (error) throw new Error(error.message);
    await audit(banned ? "user.disable" : "user.enable", "user", id);
    done();
    return null;
  });
}

export async function deleteUser(id: string): Promise<ActionResult> {
  return run(async () => {
    const { admin, viewer } = await requireAdminStrict();
    if (id === viewer.userId) throw new Error("self_delete");
    // ban first so no refresh can happen; access tokens already issued stay valid until they expire (≤ 1 h)
    await admin.auth.admin.updateUserById(id, { ban_duration: "876000h" });
    const { error } = await admin.auth.admin.deleteUser(id);
    if (error) throw new Error(error.message);
    await audit("user.delete", "user", id);
    done();
    return null;
  });
}
