from fastapi.testclient import TestClient
from backend.main import app
from unittest.mock import patch

client = TestClient(app)

def test_health():
    response = client.get('/health')

    assert response.status_code == 200
    assert response.json() == {
        'status': 'OK',
        'message': 'Research Agent API is running'
    }

@patch('backend.main.run_pipeline')
def test_research(mock_run_pipeline):
    mock_run_pipeline.return_value = {
        'query': 'What is RAG?',
        'final_answer': 'RAG combines retrieval and generation'
    }

    response = client.post(
        '/research',
        json={
            'query': 'What is RAG?',
            'chat_history': [],
            'use_planner': False,
            'use_critic': False
        }
    )

    assert response.status_code == 200
    assert response.json() == {
        'query': 'What is RAG?',
        'final_answer': 'RAG combines retrieval and generation'
    }
    mock_run_pipeline.assert_called_once_with(
        "What is RAG?",
        [],
        False,
        False
    )



def test_research_invalid_chat_role():
    response = client.post(
        '/research',
        json={
            'query': 'What is RAG?',
            'chat_history': [
                {
                    'role': 'admin',
                    'content': 'What is RAG?'
                }
            ],
            'use_planner': False,
            'use_critic': False
        }
    )

    assert response.status_code == 422

@patch('backend.main.run_pipeline')
def test_invalid_request_does_not_call_pipeline(mock_run_pipeline):
    response = client.post(
        '/research',
        json={
            'query': 'What is RAG?',
            'chat_history': [
                {
                    'role': 'admin',
                    'content': 'What is RAG?'
                }
            ],
            'use_planner': False,
            'use_critic': False
        }
    )


    assert response.status_code == 422
    mock_run_pipeline.assert_not_called()

def test_research_missing_query():
    response = client.post(
        '/research',
        json={
            'chat_history': [],
            'use_planner': False,
            'use_critic': False
        }
    )

    assert response.status_code == 422


@patch('backend.main.run_pipeline')
def research_pipeline_failure(mock_run_pipeline):
    mock_run_pipeline.side_effect = RuntimeError('Internal details')
    response = client.post(
        '/research',
        json={
            'query': 'What is RAG?',
            'chat_history': [],
            'use_planner': False,
            'use_critic': False
        }
    )

    assert response.status_code == 500
    assert response.json() == {
        'detail': 'Research pipeline failed'
    }