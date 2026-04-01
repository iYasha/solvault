class TestMessagesEndpoint:
    """Messages endpoint should accept, route, and generate agent responses."""

    async def test_creates_message(self, client):
        """Given a valid payload, when POST /v1/messages, then 200 with session_id, message_id, and response_text."""
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
        assert "response_text" in data, "Response should contain response_text"

    async def test_returns_agent_response(self, client):
        """Given a valid message, when POST /v1/messages, then response contains agent's response text."""
        response = await client.post(
            "/v1/messages",
            json={
                "channel": "cli",
                "user_id": "testuser",
                "text": "hello sol",
            },
        )

        data = response.json()
        assert data["response_text"] == "Test response from Sol", "Should return the agent's response"

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


class TestGetHistory:
    """History endpoint should return full chat history for a session."""

    async def test_returns_empty_history_for_new_user(self, client):
        """Given no prior messages, when GET history, then empty messages list is returned."""
        response = await client.get("/v1/messages/cli/newuser/history")

        assert response.status_code == 200, "Should return 200 for new user"
        data = response.json()
        assert data["messages"] == [], "Should return empty messages list"
        assert "session_id" in data, "Should contain session_id"

    async def test_returns_messages_after_conversation(self, client):
        """Given prior messages, when GET history, then all messages are returned in order."""
        await client.post("/v1/messages", json={"channel": "cli", "user_id": "histuser", "text": "first"})
        await client.post("/v1/messages", json={"channel": "cli", "user_id": "histuser", "text": "second"})

        response = await client.get("/v1/messages/cli/histuser/history")

        data = response.json()
        messages = data["messages"]
        assert len(messages) == 4, "Should have 2 user + 2 assistant messages"
        assert messages[0]["role"] == "user", "First message should be from user"
        assert messages[0]["content"] == "first", "First message content should match"
        assert messages[1]["role"] == "assistant", "Second message should be from assistant"
        assert messages[2]["role"] == "user", "Third message should be from user"
        assert messages[2]["content"] == "second", "Third message content should match"

    async def test_rejects_invalid_channel(self, client):
        """Given an invalid channel, when GET history, then 422 validation error."""
        response = await client.get("/v1/messages/invalid_channel/testuser/history")

        assert response.status_code == 422, "Should reject invalid channel with 422"
