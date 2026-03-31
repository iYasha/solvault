from sol import __version__


class TestHealthEndpoint:
    """Health endpoint should return server status and version."""

    async def test_returns_ok(self, client):
        """Given a running server, when GET /v1/health, then 200 with status=ok and version."""
        response = await client.get("/v1/health")

        assert response.status_code == 200, "Health endpoint should return 200"
        data = response.json()
        assert data["status"] == "ok", "Status should be 'ok'"
        assert data["version"] == __version__, "Version should match package version"
