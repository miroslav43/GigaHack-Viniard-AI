"use client";

// State of the live cadastre layer (src/lib/cadastre.ts): the switch (off by default: it calls an external server)
// and the last parcel query of a map click. A new query cancels the one in flight.
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchParcel, type CadastreParcel } from "@/lib/cadastre";

export type CadastreQuery =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "found"; parcel: CadastreParcel }
  | { status: "none" }
  | { status: "error"; detail: string };

export interface Cadastre {
  on: boolean;
  setOn: (on: boolean) => void;
  query: CadastreQuery;
  pick: (lon: number, lat: number) => void;
  clear: () => void;
}

const IDLE: CadastreQuery = { status: "idle" };

export function useCadastre(): Cadastre {
  const [on, setOnState] = useState(false);
  const [query, setQuery] = useState<CadastreQuery>(IDLE);
  const inFlight = useRef<AbortController | null>(null);

  const clear = useCallback(() => {
    inFlight.current?.abort();
    inFlight.current = null;
    setQuery(IDLE);
  }, []);

  const pick = useCallback((lon: number, lat: number) => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setQuery({ status: "loading" });
    fetchParcel(lon, lat, controller.signal).then(
      (parcel) => {
        if (inFlight.current === controller) setQuery(parcel ? { status: "found", parcel } : { status: "none" });
      },
      (e: unknown) => {
        // a newer click (or closing the panel) aborted this one: nothing to report
        if (inFlight.current !== controller) return;
        const timedOut = e instanceof DOMException && e.name === "TimeoutError";
        setQuery({ status: "error", detail: timedOut ? "timeout" : e instanceof Error ? e.message : String(e) });
      },
    );
  }, []);

  const setOn = useCallback(
    (next: boolean) => {
      setOnState(next);
      if (!next) clear();
    },
    [clear],
  );

  useEffect(() => () => inFlight.current?.abort(), []);

  return { on, setOn, query, pick, clear };
}
