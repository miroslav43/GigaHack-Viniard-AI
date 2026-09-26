"use client";

import Button, { type ButtonProps } from "@mui/material/Button";
import { useRouter } from "@/i18n/routing";
import { DEMO_COOKIE } from "@/lib/supabase/config";

/** Enters the public demo (read-only Sireț3 data): the same cookie as "Demo fără cont" on the login page. */
export function DemoButton({ children, ...props }: ButtonProps) {
  const router = useRouter();
  const start = () => {
    document.cookie = `${DEMO_COOKIE}=1; Max-Age=${60 * 60 * 24 * 7}; path=/; SameSite=Lax`;
    router.push("/");
    router.refresh();
  };
  return (
    <Button {...props} onClick={start}>
      {children}
    </Button>
  );
}
