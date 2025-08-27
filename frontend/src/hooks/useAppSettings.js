import { useEffect, useState } from "react";

const KEY = "appSettings";

export function useAppSettings() {
  const [settings, setSettings] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(KEY)) || { refreshMs: 30000 };
    } catch {
      return { refreshMs: 30000 };
    }
  });

  useEffect(() => {
    localStorage.setItem(KEY, JSON.stringify(settings));
  }, [settings]);

  return { settings, setSettings };
}

export default useAppSettings;
