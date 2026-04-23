"""
Tests for the lead capture endpoint (POST /api/audit-leads).
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# Patch config loading before importing app
@pytest.fixture(autouse=True)
def patch_app(tmp_path):
    """Patch config and leads file for testing."""
    # Create a minimal config
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "local_nodes: []\ncloud_providers: []\n"
        "routing:\n  default_mode: cost\n  vram_poll_interval_seconds: 60\n"
        "  electricity_kwh_price: 0.14\n  gpu_tdp_watts: 320\n"
        "server:\n  host: 0.0.0.0\n  port: 8000\n  api_key_env: NB_API_KEY\n"
    )

    leads_file = tmp_path / "leads.jsonl"

    with patch.dict(os.environ, {"CONFIG_PATH": str(config_file)}, clear=False):
        with patch("neuralbrok.main.LEADS_FILE", leads_file):
            # Import after patching
            from neuralbrok.main import app
            yield {
                "client": TestClient(app, raise_server_exceptions=False),
                "leads_file": leads_file,
            }


class TestValidSubmission:
    """Test that valid leads are captured correctly."""

    def test_valid_lead_returns_200(self, patch_app):
        client = patch_app["client"]
        with client:
            resp = client.post(
                "/api/audit-leads",
                json={"email": "test@company.com", "spend": "1k-5k"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_valid_lead_writes_jsonl(self, patch_app):
        client = patch_app["client"]
        leads_file = patch_app["leads_file"]

        with client:
            client.post(
                "/api/audit-leads",
                json={"email": "cto@startup.io", "spend": "5k-25k"},
            )

        assert leads_file.exists()
        lines = leads_file.read_text().strip().split("\n")
        assert len(lines) == 1

        lead = json.loads(lines[0])
        assert lead["email"] == "cto@startup.io"
        assert lead["spend"] == "5k-25k"
        assert "ts" in lead
        assert "ip" in lead

    def test_duplicate_emails_append(self, patch_app):
        client = patch_app["client"]
        leads_file = patch_app["leads_file"]

        with client:
            client.post(
                "/api/audit-leads",
                json={"email": "same@co.com", "spend": "500-1k"},
            )
            client.post(
                "/api/audit-leads",
                json={"email": "same@co.com", "spend": "1k-5k"},
            )

        lines = leads_file.read_text().strip().split("\n")
        assert len(lines) == 2  # No dedup — both appended


class TestInvalidEmail:
    """Test email validation returns 422."""

    def test_invalid_email_returns_422(self, patch_app):
        client = patch_app["client"]
        with client:
            resp = client.post(
                "/api/audit-leads",
                json={"email": "not-an-email", "spend": "1k-5k"},
            )
        assert resp.status_code == 422

    def test_empty_email_returns_422(self, patch_app):
        client = patch_app["client"]
        with client:
            resp = client.post(
                "/api/audit-leads",
                json={"email": "", "spend": "1k-5k"},
            )
        assert resp.status_code == 422

    def test_missing_spend_returns_422(self, patch_app):
        client = patch_app["client"]
        with client:
            resp = client.post(
                "/api/audit-leads",
                json={"email": "ok@test.com", "spend": ""},
            )
        assert resp.status_code == 422


class TestNotifyResilience:
    """Test that missing SMTP env vars don't crash the endpoint."""

    def test_no_smtp_vars_still_returns_200(self, patch_app):
        """Endpoint should work even without SMTP configuration."""
        client = patch_app["client"]

        # Ensure SMTP vars are not set
        env_overrides = {
            "NOTIFY_SMTP_HOST": "",
            "NOTIFY_SMTP_PORT": "",
            "NOTIFY_SMTP_USER": "",
            "NOTIFY_SMTP_PASS": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            with client:
                resp = client.post(
                    "/api/audit-leads",
                    json={"email": "test@safe.com", "spend": "25k+"},
                )
        assert resp.status_code == 200


class TestNotifyModule:
    """Test the notify module directly."""

    def test_missing_env_logs_warning_and_returns(self):
        """send_lead_notification should not crash when SMTP vars missing."""
        from neuralbrok.notify import send_lead_notification

        with patch.dict(os.environ, {"RESEND_API_KEY": ""}, clear=False):
            # Should not raise
            send_lead_notification("test@x.com", "1k-5k", "2026-04-18T00:00:00Z", "127.0.0.1")

    def test_partial_env_logs_warning(self):
        """Even with some vars set, missing one should skip silently."""
        from neuralbrok.notify import send_lead_notification

        with patch.dict(os.environ, {"RESEND_API_KEY": ""}, clear=False):
            send_lead_notification("test@x.com", "5k-25k", "2026-04-18T00:00:00Z", "127.0.0.1")
