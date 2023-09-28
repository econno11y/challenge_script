import pytest

from challenge.script import get_llm_caption, get_nasa_image

from .mock_data import bad_nasa_response, llm_response, nasa_response


class MockResponse:
    def __init__(self, json_data, status_code, text=""):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.json_data


class MockReplicateClient:
    def __init__(self, api_token):
        self.api_token = api_token

    def run(self, model, input):
        return llm_response


@pytest.fixture
def mock_requests_get_200(monkeypatch):
    def mock_get(url):
        mock = MockResponse(nasa_response, status_code=200)
        return mock

    monkeypatch.setattr("requests.get", mock_get)


@pytest.fixture
def mock_request_get_404(monkeypatch):
    def mock_get(url):
        mock = MockResponse(json_data="", text="Error message", status_code=404)
        return mock

    monkeypatch.setattr("requests.get", mock_get)


@pytest.fixture
def mock_replicate_run(monkeypatch):
    monkeypatch.setattr("replicate.Client", MockReplicateClient)


def test_get_nasa_image_200(mock_requests_get_200):
        response = get_nasa_image()
        assert response == nasa_response

def test_get_nasa_image_404(mock_request_get_404):
    with pytest.raises(
        Exception, match="Error fetching image from NASA API: Error message"
    ):
        get_nasa_image()


def test_get_llm_caption(mock_replicate_run):
    with pytest.raises(Exception):
        get_llm_caption(bad_nasa_response.get("url"))

    response = get_llm_caption(nasa_response.get("url"))
    assert response == llm_response
