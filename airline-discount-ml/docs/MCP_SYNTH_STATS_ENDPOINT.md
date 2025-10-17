# MCP Synth Stats Endpoint Implementation

## Overview
Added `synth_stats` endpoint to the MCP Synth server to provide file statistics for generated synthetic data.

## Endpoint Details

### Request Model: `SynthStatsRequest`
```python
class SynthStatsRequest(BaseModel):
    data_dir: str = Field(default="data/synthetic_output")
    log_file: str = Field(default="")
```

### Response Model: `SynthStatsResponse`
```python
class SynthStatsResponse(BaseModel):
    data_dir: str
    files: List[str]
    total_files: int
    total_size_bytes: int
    stats_by_file: Dict[str, Dict[str, Any]]
```

## Features

1. **Directory Scanning**
   - Recursively finds all files in `data_dir`
   - Returns sorted file list
   - Validates directory exists

2. **Size Calculation**
   - Total size in bytes
   - Individual file sizes
   - Formatted output in MB

3. **Record Counting**
   - JSON arrays: Parses and counts elements
   - NDJSON: Counts non-empty lines
   - CSV: Counts rows using DictReader
   - Graceful handling of unparseable files

4. **Output Formatting**
   - Emoji-enhanced console output (📅 📁 💾 📄)
   - Human-readable byte sizes
   - Comma-separated thousands
   - Record counts when available

5. **Persistence**
   - Saves statistics to `data_statistics.txt` by default
   - Custom log file path support
   - Creates output directory if needed

## MCP Integration

### JSON Schema
```python
SYNTH_STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "data_dir": {"type": "string", "default": "data/synthetic_output"},
        "log_file": {"type": "string", "default": ""}
    },
    "required": []
}
```

### Tools List Entry
```python
{
    "name": "synth_stats",
    "description": "Get statistics about generated synthetic data files (size, record counts)",
    "input_schema": SYNTH_STATS_SCHEMA
}
```

### Tool Call Handler
```python
elif name == "synth_stats":
    req = SynthStatsRequest(**args)
    resp = synth_stats(req)
    # Format and return statistics
```

## Example Output

```
📅 Statistics for C:\repos\copilot\airline-discount-ml\data\synthetic_output

📁 Total files: 3
💾 Total size: 124,567 bytes (0.12 MB)

📄 File Details:
  - generated_data.json: 98,234 bytes, 1,000 records
  - generation_log.txt: 15,123 bytes
  - model_inspection.txt: 11,210 bytes

💾 Output also saved to: data/synthetic_output/data_statistics.txt
```

## Error Handling

- **404 HTTPException**: If `data_dir` doesn't exist or isn't a directory
- **Silent failures**: Unparseable files don't crash (record_count = None)
- **Encoding safety**: UTF-8 encoding specified for all file operations

## Usage

### Via MCP JSON-RPC
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "synth_stats",
    "arguments": {
      "data_dir": "data/synthetic_output"
    }
  }
}
```

### Direct Function Call
```python
from src.mcp_synth.server import synth_stats, SynthStatsRequest

req = SynthStatsRequest(data_dir="data/synthetic_output")
result = synth_stats(req)
print(f"Total files: {result.total_files}")
print(f"Total size: {result.total_size_bytes} bytes")
```

## Integration with VS Code MCP

This endpoint is registered in `.vscode/mcp.json`:
```json
{
  "servers": {
    "synth-local": {
      "toolAllowList": [
        "synth_generate",
        "synth_inspect_model",
        "preview_table_head",
        "synth_stats"  ← Added in PR #2
      ]
    }
  }
}
```

## Benefits

1. **Data Visibility**: Quickly see what Synth generated without manual file inspection
2. **Quality Checks**: Verify record counts match expectations
3. **Storage Management**: Monitor synthetic data storage usage
4. **Debugging**: Confirm files were created and are parseable
5. **Documentation**: Auto-generated statistics for training documentation

## Related Files

- `.vscode/mcp.json` - Tool allowlist registration
- `airline-discount-ml/src/mcp_synth/server.py` - Implementation
- `airline-discount-ml/tests/mcp_synth/test_server.py` - Tests for synth_stats

## Testing

See `tests/mcp_synth/test_server.py`:
- `test_stats_with_files()` ✅
- `test_stats_empty_directory()` ✅
- `test_stats_nonexistent_directory()` ✅
