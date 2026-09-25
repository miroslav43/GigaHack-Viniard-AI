"use client";

import { useMemo } from "react";
import { useLocale } from "next-intl";
import { makeFormat } from "./format";

export function useFormat() {
  const locale = useLocale();
  return useMemo(() => makeFormat(locale), [locale]);
}
