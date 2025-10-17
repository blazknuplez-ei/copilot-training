"""Tests for MCP Synth server module."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.mcp_synth.server import (
    app,
    synth_inspect_model,
    preview_table_head,
    synth_stats,
    InspectModelRequest,
    PreviewHeadRequest,
    SynthStatsRequest
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and version endpoints."""
    
    def test_healthz_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_version_endpoint(self, client):
        """Test version endpoint."""
        response = client.get("/version")
        assert response.status_code == 200
        assert "version" in response.json()


class TestSynthGenerate:
    """Tests for synth_generate endpoint."""
    
    @patch('src.mcp_synth.server.subprocess.run')
    def test_synth_generate_success(self, mock_run, client, tmp_path):
        """Test successful data generation."""
        # Mock subprocess output
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "passengers": [{"id": 1, "name": "Alice"}],
            "routes": [{"id": 1, "origin": "NYC"}]
        })
        mock_run.return_value = mock_result
        
        response = client.post("/synth_generate", json={
            "model_dir": "synth_models/airline_data",
            "out_dir": str(tmp_path),
            "size": 100,
            "seed": 42
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "files_created" in data
    
    @patch('src.mcp_synth.server.subprocess.run')
    def test_synth_generate_with_custom_size(self, mock_run, client, tmp_path):
        """Test generation with custom size parameter."""
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"data": []})
        mock_run.return_value = mock_result
        
        response = client.post("/synth_generate", json={
            "model_dir": "synth_models/airline_data",
            "out_dir": str(tmp_path),
            "size": 500,
            "seed": 42
        })
        
        assert response.status_code == 200
        # Verify size was passed to subprocess
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "--size" in call_args
        assert "500" in call_args


class TestSynthInspectModel:
    """Tests for synth_inspect_model function."""
    
    def test_inspect_model_with_valid_directory(self, tmp_path):
        """Test inspecting a valid model directory."""
        # Create some test files
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "schema.json").write_text("{}")
        (model_dir / "config.json").write_text("{}")
        
        req = InspectModelRequest(model_dir=str(model_dir))
        result = synth_inspect_model(req)
        
        assert result.model_dir == str(model_dir)
        assert len(result.files) == 2
        assert any("schema.json" in f for f in result.files)
    
    def test_inspect_model_with_nonexistent_directory(self, tmp_path):
        """Test inspecting a non-existent directory."""
        req = InspectModelRequest(model_dir=str(tmp_path / "nonexistent"))
        
        with pytest.raises(Exception):  # Should raise HTTPException
            synth_inspect_model(req)
    
    def test_inspect_model_empty_directory(self, tmp_path):
        """Test inspecting an empty directory."""
        model_dir = tmp_path / "empty"
        model_dir.mkdir()
        
        req = InspectModelRequest(model_dir=str(model_dir))
        result = synth_inspect_model(req)
        
        assert result.model_dir == str(model_dir)
        assert len(result.files) == 0


class TestPreviewTableHead:
    """Tests for preview_table_head function."""
    
    def test_preview_json_array(self, tmp_path):
        """Test previewing a JSON array file."""
        # Create file in allowed directory (data/)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        data_file = data_dir / "test_preview.json"
        
        test_data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"}
        ]
        data_file.write_text(json.dumps(test_data))
        
        try:
            req = PreviewHeadRequest(path=str(data_file), n=2)
            result = preview_table_head(req)
            
            assert len(result.rows) == 2
            assert result.rows[0]["name"] == "Alice"
        finally:
            # Cleanup
            if data_file.exists():
                data_file.unlink()
    
    def test_preview_csv_file(self, tmp_path):
        """Test previewing a CSV file."""
        # Create file in allowed directory (data/)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        data_file = data_dir / "test_preview.csv"
        
        csv_content = "id,name\n1,Alice\n2,Bob\n3,Charlie"
        data_file.write_text(csv_content)
        
        try:
            req = PreviewHeadRequest(path=str(data_file), n=2)
            result = preview_table_head(req)
            
            assert len(result.rows) == 2
            assert result.rows[0]["name"] == "Alice"
        finally:
            # Cleanup
            if data_file.exists():
                data_file.unlink()
    
    def test_preview_limits_rows(self, tmp_path):
        """Test that preview respects the n parameter."""
        # Create file in allowed directory (data/)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        data_file = data_dir / "test_preview_limits.json"
        
        test_data = [{"id": i} for i in range(100)]
        data_file.write_text(json.dumps(test_data))
        
        try:
            req = PreviewHeadRequest(path=str(data_file), n=10)
            result = preview_table_head(req)
            
            assert len(result.rows) == 10
        finally:
            # Cleanup
            if data_file.exists():
                data_file.unlink()


class TestSynthStats:
    """Tests for synth_stats function."""
    
    def test_stats_with_files(self, tmp_path):
        """Test statistics calculation with files."""
        # Create test files
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "data.json").write_text('[{"id": 1}, {"id": 2}]')
        
        req = SynthStatsRequest(out_dir=str(out_dir))
        result = synth_stats(req)
        
        assert result.out_dir == str(out_dir)
        assert len(result.files) > 0
    
    def test_stats_empty_directory(self, tmp_path):
        """Test statistics with empty directory."""
        out_dir = tmp_path / "empty"
        out_dir.mkdir()
        
        req = SynthStatsRequest(out_dir=str(out_dir))
        result = synth_stats(req)
        
        assert result.out_dir == str(out_dir)
        assert len(result.files) == 0
    
    def test_stats_nonexistent_directory(self, tmp_path):
        """Test statistics with non-existent directory."""
        req = SynthStatsRequest(out_dir=str(tmp_path / "nonexistent"))
        
        with pytest.raises(Exception):  # Should raise HTTPException
            synth_stats(req)
