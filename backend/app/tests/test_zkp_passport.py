from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.zkp_passport import router


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def cache_set(self, key, value, ttl=0):
        self.store[key] = value

    async def cache_get(self, key):
        return self.store.get(key)


def build_client():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.redis = FakeRedis()
    return TestClient(app)


def test_request_and_verify_passport():
    client = build_client()

    resp = client.post('/api/v1/zkp/passport/request', json={'buyer_id': 'test-buyer', 'threshold_usd': 5000})
    assert resp.status_code == 200
    data = resp.json()
    assert 'token' in data and data['verified'] is True

    verify = client.post('/api/v1/zkp/passport/verify', json={'token': data['token']})
    assert verify.status_code == 200
    v = verify.json()
    assert v.get('verified') is True
