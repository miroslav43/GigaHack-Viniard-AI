// Account emails sent by Supabase Auth: invitations and password resets. Nobody sets a password for someone
// else — the link lands on /parola-noua, where the account owner chooses it (SetPasswordForm).
import "server-only";
import { headers } from "next/headers";
import { getLocale } from "next-intl/server";
import type { AuthError, User } from "@supabase/supabase-js";
import { routing } from "@/i18n/routing";
import type { createAdminClient } from "./supabase/admin";

type Admin = ReturnType<typeof createAdminClient>;

/** Absolute URL of the set-password page, on the host the admin is using (unless SITE_URL pins it).
 *  Supabase only honours it if it is in Auth → URL Configuration → Redirect URLs; otherwise it falls back to the Site URL. */
async function setPasswordUrl() {
  const locale = await getLocale();
  const path = `${locale === routing.defaultLocale ? "" : `/${locale}`}/parola-noua`;
  if (process.env.SITE_URL) return `${process.env.SITE_URL.replace(/\/$/, "")}${path}`;
  const h = await headers();
  const host = h.get("x-forwarded-host") ?? h.get("host") ?? "localhost:3000";
  const proto = h.get("x-forwarded-proto") ?? (/^(localhost|127\.)/.test(host) ? "http" : "https");
  return `${proto}://${host}${path}`;
}

/** Auth error → an error code the admin panels translate (superAdmin.errors.*). */
const emailError = (e: AuthError) =>
  e.code === "email_exists" || e.code === "user_already_exists"
    ? "email_exists"
    : e.code === "over_email_send_rate_limit" || e.status === 429
      ? "email_rate_limit"
      : e.code === "email_address_not_authorized"
        ? "email_not_authorized"
        : e.code === "email_address_invalid"
          ? "email_undeliverable"
          : e.message;

/** Has the account never set a password (invited, link not used yet)? */
export const isPendingInvite = (u: User) => Boolean(u.invited_at && !u.email_confirmed_at);

/** Creates the account and emails the invitation; municipality and role are app_metadata (server-set). */
export async function inviteUser(admin: Admin, email: string, fullName: string, appMetadata: { uat: string | null; uat_role: string }) {
  const { data, error } = await admin.auth.admin.inviteUserByEmail(email, {
    data: { full_name: fullName.trim() || undefined },
    redirectTo: await setPasswordUrl(),
  });
  if (error) throw new Error(emailError(error));
  // the invite API only takes user_metadata
  const { error: metaError } = await admin.auth.admin.updateUserById(data.user.id, { app_metadata: appMetadata });
  if (metaError) {
    await admin.auth.admin.deleteUser(data.user.id);
    throw new Error(metaError.message);
  }
  return data.user;
}

/** Emails a new link: the invitation again while it is pending (links expire), otherwise a password reset. */
export async function sendPasswordLink(admin: Admin, user: User): Promise<"invite" | "reset"> {
  if (!user.email) throw new Error("invalid_email");
  const redirectTo = await setPasswordUrl();
  if (isPendingInvite(user)) {
    const { error } = await admin.auth.admin.inviteUserByEmail(user.email, { redirectTo });
    if (error) throw new Error(emailError(error));
    return "invite";
  }
  // implicit flow (the admin client's default): the session comes back in the URL fragment, as for invitations
  const { error } = await admin.auth.resetPasswordForEmail(user.email, { redirectTo });
  if (error) throw new Error(emailError(error));
  return "reset";
}
