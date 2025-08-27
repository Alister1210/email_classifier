import React, { useEffect } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { useAppSettings } from "../hooks/useAppSettings";
import { useGmailConnection } from "../hooks/useGmailConnection";
import { AuthService } from "../services/api";
import { useNavigate } from "react-router-dom";

export default function Settings() {
  const { settings, setSettings } = useAppSettings();
  const { isConnected, email, connect, disconnect } = useGmailConnection();
  const navigate = useNavigate();

  // Try to fetch profile if connected but email is missing
  useEffect(() => {
    async function fetchProfile() {
      if (isConnected && !email) {
        try {
          const profile = await AuthService.getProfile();
          if (profile?.email) {
            connect(profile.email);
          }
        } catch (e) {
          console.error("Failed to fetch profile:", e);
        }
      }
    }
    fetchProfile();
  }, [isConnected, email, connect]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Refresh interval */}
        <section className="space-y-2">
          <div className="text-sm font-medium">Refresh interval</div>
          <select
            className="h-9 rounded-md border border-[hsl(var(--border))] bg-transparent px-3 text-sm"
            value={settings.refreshMs}
            onChange={(e) =>
              setSettings({ ...settings, refreshMs: Number(e.target.value) })
            }
          >
            <option value={15000}>15 seconds</option>
            <option value={30000}>30 seconds</option>
            <option value={60000}>1 minute</option>
            <option value={120000}>2 minutes</option>
          </select>
        </section>

        {/* Gmail account */}
        <section className="space-y-2">
          <div className="text-sm font-medium">Gmail account</div>
          <div className="flex items-center gap-2">
            <Button
              onClick={async () => {
                try {
                  const data = await AuthService.login();
                  window.location.href = data.authorization_url;
                } catch (e) {
                  console.error(e);
                }
              }}
            >
              {isConnected ? "Change account" : "Connect account"}
            </Button>
            {isConnected && (
              <Button
                variant="secondary"
                onClick={() => {
                  disconnect();
                  navigate("/");
                }}
              >
                Disconnect
              </Button>
            )}
          </div>

          {/* Auto-detected connected email */}
          {isConnected && email && (
            <div className="text-sm text-muted-foreground">
              Connected as: <span className="font-medium">{email}</span>
            </div>
          )}
        </section>
      </CardContent>
    </Card>
  );
}
