import pytest
from unittest.mock import Mock
from agents.research_agent import run_agent


def test_run_agent_propogates_exception():
    agent_excecutor = Mock()
    agent_excecutor.invoke.side_effect = RuntimeError('Agent Failed')


    with pytest.raises(RuntimeError, match='Agent Failed'):
        run_agent(
            query='What is RAG?',
            agent_executor=agent_excecutor
        )