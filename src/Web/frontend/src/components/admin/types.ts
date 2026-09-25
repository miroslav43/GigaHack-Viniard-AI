import type { Geometry } from "geojson";

// shared by server and client modules (a "use client" module cannot export plain values to the server)
export type AdminTab = "overview" | "uat" | "users" | "surveys" | "system" | "audit";
export const ADMIN_TABS: AdminTab[] = ["overview", "uat", "users", "surveys", "system", "audit"];

export type Country = "MD" | "RO";
export type Role = "platform_admin" | "uat_admin" | "inspector" | "viewer";
export const ROLES: Role[] = ["platform_admin", "uat_admin", "inspector", "viewer"];

export interface AdminUat {
  key: string;
  name: string;
  district: string | null;
  country: Country;
  osm_relation_id: number | null;
  area_ha: number;
  active: boolean;
  created_at: string;
  geofence: Geometry;
  surveys: { id: string; overlap_ha: number }[];
  /** null when the Admin API (secret key) is not configured */
  users: number | null;
}

export interface AdminUser {
  id: string;
  email: string;
  name: string | null;
  uat: string | null;
  role: Role | null;
  created_at: string;
  last_sign_in_at: string | null;
  banned: boolean;
}

export interface AdminSurvey {
  id: string;
  name: string;
  captured_at: string | null;
  area_ha: number;
  data_path: string;
  uats: { key: string; name: string; overlap_ha: number }[];
}

export interface LocalBundle {
  id: string;
  name: string;
  registered: boolean;
}
