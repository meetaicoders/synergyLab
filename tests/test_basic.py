"""
Basic test module for the generator package.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_import():
    """Test that the generator package can be imported."""
    try:
        import generator
        assert True
    except ImportError:
        assert False, "Failed to import generator package"

def test_modules():
    """Test that important modules are available."""
    try:
        from generator import agents, llms, tools
        assert agents
        assert llms
        assert tools
        assert True
    except ImportError as e:
        assert False, f"Failed to import modules: {e}" 