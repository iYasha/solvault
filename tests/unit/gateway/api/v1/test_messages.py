class TestMessagesEndpoint:
    """Messages endpoint should accept and route incoming messages."""

    async def test_creates_message(self, client):
        """Given a valid payload, when POST /v1/messages, then 200 with session_id and message_id."""
        response = await client.post(
            "/v1/messages",
            json={
                "channel": "cli",
                "user_id": "testuser",
                "text": "hello sol",
            },
        )

        assert response.status_code == 200, "Should accept valid message"
        data = response.json()
        assert "session_id" in data, "Response should contain session_id"
        assert "message_id" in data, "Response should contain message_id"

    async def test_rejects_invalid_channel(self, client):
        """Given an invalid channel, when POST /v1/messages, then 422 validation error."""
        response = await client.post(
            "/v1/messages",
            json={
                "channel": "invalid_channel",
                "user_id": "testuser",
                "text": "hello",
            },
        )

        assert response.status_code == 422, "Should reject invalid channel with 422"

    async def test_reuses_session_for_same_user(self, client):
        """Given two messages from same user, when posted, then same session_id is returned."""
        payload = {"channel": "cli", "user_id": "testuser", "text": "hello"}

        resp1 = await client.post("/v1/messages", json=payload)
        resp2 = await client.post("/v1/messages", json={**payload, "text": "second"})

        assert resp1.json()["session_id"] == resp2.json()["session_id"], "Same user should get the same session"
