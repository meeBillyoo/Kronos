# Kronos Web UI API 技术文档

## 概述

Kronos Web UI 是一个基于 Flask 的金融数据预测 Web 应用程序，提供了完整的 RESTful API 接口用于数据管理、模型加载和金融预测功能。

**基础信息：**
- 服务地址：`http://localhost:7070`
- 框架：Flask + CORS
- 数据格式：JSON
- 支持的数据文件格式：CSV、Feather

## API 响应格式

所有 API 响应都遵循统一的 JSON 格式：

**成功响应：**
```json
{
  "code": 0,
  "message": "Success",
  "data": {}
}
```

**错误响应：**
```json
{
  "code": {错误码},
  "message": "错误描述",
  "data": null
}
```

**错误码说明：**
- `400`: 请求参数错误
- `500`: 服务器内部错误

## API 端点详情

### 1. 首页

**GET** `/`

返回 Web UI 主页面。

**响应：** HTML 页面

---

### 2. 获取数据文件列表

**GET** `/api/data-files`

获取 `data` 目录下所有可用的数据文件列表。

**响应示例：**
```json
{
  "code": 0,
  "message": "Success",
  "data": [
    {
      "name": "btc_data.csv",
      "path": "/path/to/data/btc_data.csv",
      "size": "1.2 MB"
    }
  ]
}
```

**数据字段说明：**
- `name`: 文件名
- `path`: 文件完整路径
- `size`: 文件大小（KB/MB）

---

### 3. 加载数据文件

**POST** `/api/load-data`

加载指定的数据文件并返回数据信息。

**请求参数：**
```json
{
  "file_path": "string" // 必需，数据文件路径
}
```

**响应示例：**
```json
{
  "success": true,
  "data_info": {
    "rows": 10000,
    "columns": ["timestamps", "open", "high", "low", "close", "volume"],
    "dtypes": {
      "open": "float64",
      "high": "float64",
      "low": "float64",
      "close": "float64",
      "volume": "float64"
    }
  },
  "message": "Successfully loaded data, total 10000 rows"
}
```

**数据要求：**
- 必需列：`open`, `high`, `low`, `close`
- 可选列：`volume`, `amount`
- 时间列：`timestamps`, `timestamp`, 或 `date`（自动处理）

---

### 4. 执行预测（文件数据输入）

**POST** `/api/predict`

使用 Kronos 模型对已加载的文件数据进行预测。

**请求参数：**
```json
{
  "file_path": "string",      // 必需，数据文件路径
  "lookback": 400,            // 可选，历史数据回看长度，默认400
  "pred_len": 120,            // 可选，预测长度，默认120
  "temperature": 0.7,         // 可选，预测温度参数，默认0.7
  "top_p": 0.9,              // 可选，Top-p 采样参数，默认0.9
  "sample_count": 1,          // 可选，采样次数，默认1
  "start_date": "2024-01-01T00:00:00" // 可选，自定义预测起始时间
}
```

**响应示例：**
```json
{
  "success": true,
  "prediction_type": "Kronos model prediction (latest data)",
  "chart": "{plotly_chart_json}",
  "prediction_results": [
    {
      "timestamp": 1640995200,
      "datetime": "2022-01-01 01:00:00",
      "open": 45000.0,
      "high": 45500.0,
      "low": 44800.0,
      "close": 45200.0,
      "volume": 1000.0,
      "amount": 0
    }
  ],
  "actual_data": [
    {
      "timestamp": 1640991600,
      "datetime": "2022-01-01 00:00:00",
      "open": 44800.0,
      "high": 45200.0,
      "low": 44600.0,
      "close": 45000.0,
      "volume": 950.0,
      "amount": 0
    }
  ],
  "has_comparison": true,
  "message": "预测完成，生成了 120 个预测点，包含 50 个实际数据点用于对比"
}
```

**功能特性：**
- 支持自定义时间窗口预测
- 自动生成 Plotly 可视化图表
- 提供预测结果与实际数据对比
- 自动保存预测结果到文件
- 预测结果和实际数据都包含秒级时间戳和格式化时间

---

### 5. 执行预测（直接K线数据输入）

**POST** `/api/predictkline`

使用 Kronos 模型对直接输入的K线数据进行预测，无需预先加载文件。

**请求参数：**
```json
{
  "kline_data": [                 // 必需，K线数据数组
    {
      "timestamp": 1640995200,    // 可选，秒级时间戳
      "datetime": "2022-01-01 00:00:00", // 可选，格式化时间
      "time": 1640995200,         // 可选，秒级时间戳（备选字段）
      "open": 47000.0,            // 必需，开盘价
      "high": 47500.0,            // 必需，最高价
      "low": 46800.0,             // 必需，最低价
      "close": 47200.0,           // 必需，收盘价
      "volume": 1000.0,           // 可选，交易量
      "amount": 47200000.0        // 可选，成交额
    }
  ],
  "lookback": 400,                // 可选，历史数据回看长度，默认400
  "pred_len": 120,                // 可选，预测长度，默认120
  "temperature": 0.7,             // 可选，预测温度参数，默认0.7
  "top_p": 0.9,                  // 可选，Top-p 采样参数，默认0.9
  "sample_count": 1               // 可选，采样次数，默认1
}
```

**K线数据要求：**
- 必需字段：`open`, `high`, `low`, `close`
- 时间字段：`timestamp`, `datetime`, 或 `time`（可选，自动检测）
- 数据长度：至少需要 `lookback` 条记录
- 数据排序：按时间升序排列

**响应示例：**
```json
{
  "success": true,
  "prediction_type": "Kronos模型预测（直接K线输入，400个历史点）",
  "chart": "{plotly_chart_json}",
  "prediction_results": [
    {
      "timestamp": 1640995200,
      "datetime": "2022-01-01 01:00:00",
      "open": 45000.0,
      "high": 45500.0,
      "low": 44800.0,
      "close": 45200.0,
      "volume": 1000.0,
      "amount": 0
    }
  ],
  "actual_data": [],
  "has_comparison": false,
  "input_data_info": {
    "total_points": 500,
    "used_points": 400,
    "timeframe": "auto-detected",
    "price_range": {
      "min": 30000.0,
      "max": 70000.0
    }
  },
  "message": "预测完成，使用了 400 个K线数据点，生成了 120 个预测点"
}
```

**功能特性：**
- 直接接受K线数据数组，无需文件加载
- 自动检测时间戳格式和数据结构
- 使用最后 `lookback` 个数据点进行预测
- 自动生成未来时间戳
- 提供详细的输入数据信息
- 智能处理时间戳转换（支持秒级时间戳和格式化时间）
- 自动过滤无效的OHLC数据
- 预测结果包含秒级时间戳和格式化时间

**与 `/api/predict` 的区别：**
- 输入方式：直接传入K线数据 vs 文件路径
- 预处理：自动处理数据格式 vs 依赖预加载
- 时间范围：使用最新数据 vs 支持自定义时间窗口
- 实际数据对比：不提供 vs 可提供历史数据对比

---

### 6. 加载模型

**POST** `/api/load-model`

加载指定的 Kronos 预测模型。

**请求参数：**
```json
{
  "model": "kronos-base",     // 可选，模型名称，默认 "kronos-base"
  "device": "cpu"            // 可选，运行设备，默认 "cpu"
}
```

**可用模型：**
- `kronos-mini`: 轻量级模型（4.1M 参数，上下文长度 2048）
- `kronos-small`: 小型模型（24.7M 参数，上下文长度 512）
- `kronos-base`: 基础模型（102.3M 参数，上下文长度 512）

**响应示例：**
```json
{
  "code": 0,
  "message": "Model loaded successfully: Kronos-base (102.3M) on cpu",
  "data": {
    "success": true,
    "model_info": {
      "name": "Kronos-base",
      "params": "102.3M",
      "context_length": 512,
      "description": "Base model, provides better prediction quality"
    }
  }
}
```

---

### 7. 获取可用模型列表

**GET** `/api/available-models`

获取所有可用的预测模型信息。

**响应示例：**
```json
{
  "code": 0,
  "message": "Success",
  "data": {
    "models": {
      "kronos-mini": {
        "name": "Kronos-mini",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "context_length": 2048,
        "params": "4.1M",
        "description": "Lightweight model, suitable for fast prediction"
      },
      "kronos-small": {
        "name": "Kronos-small",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "24.7M",
        "description": "Small model, balanced performance and speed"
      },
      "kronos-base": {
        "name": "Kronos-base",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "102.3M",
        "description": "Base model, provides better prediction quality"
      }
    },
    "model_available": true
  }
}
```

---

### 8. 获取模型状态

**GET** `/api/model-status`

获取当前模型的加载状态和配置信息。

**响应示例：**
```json
{
  "code": 0,
  "message": "Success",
  "data": {
    "model_available": true,
    "model_loaded": true,
    "tokenizer_loaded": true,
    "predictor_ready": true,
    "current_model": {
      "device": "cpu",
      "max_context": 512
    }
  }
}
```

**状态字段说明：**
- `model_available`: 模型库是否可用
- `model_loaded`: 模型是否已加载
- `tokenizer_loaded`: 分词器是否已加载
- `predictor_ready`: 预测器是否就绪

---

### 9. 获取交易所K线数据

**POST** `/api/exchange-kline`

从指定交易所获取实时K线数据，支持 OKX 和 Binance 交易所。

**请求参数：**
```json
{
  "exchange": "okx",           // 可选，交易所名称，默认 "okx"，支持 "okx", "binance"
  "symbol": "BTC-USDT",       // 可选，交易对，默认 "BTC-USDT"
  "timeframe": "1H",          // 可选，时间周期，默认 "1H"
  "limit": 100                // 可选，数据条数，默认 100
}
```

**支持的交易所：**
- `okx`: OKX交易所（原OKEx）
- `binance`: 币安交易所

**OKX 支持的时间周期：**
- `1m`, `3m`, `5m`, `15m`, `30m`, `1H`, `2H`, `4H`, `6H`, `12H`, `1D`, `1W`, `1M`

**Binance 支持的时间周期：**
- `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`

**响应示例：**
```json
{
  "code": 0,
  "message": "K线数据获取成功",
  "data": {
    "success": true,
    "exchange": "OKX",
    "symbol": "BTC-USDT",
    "timeframe": "1H",
    "count": 100,
    "kline_data": [
      {
        "timestamp": 1640995200,
        "datetime": "2022-01-01 00:00:00",
        "open": 47000.0,
        "high": 47500.0,
        "low": 46800.0,
        "close": 47200.0,
        "volume": 1000.0,
        "amount": 47200000.0
      }
    ],
    "data_info": {
      "rows": 100,
      "columns": ["timestamp", "datetime", "open", "high", "low", "close", "volume", "amount"],
      "price_range": {
        "min": 46000.0,
        "max": 48000.0
      },
      "time_range": {
        "start": "2022-01-01T00:00:00",
        "end": "2022-01-05T03:00:00"
      }
    }
  }
}
```

**K线数据字段说明：**
- `time`: 秒级时间戳（TradingView格式）
- `datetime`: 格式化时间字符串
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 交易量
- `amount`: 成交额

**API 接口说明：**
- OKX API：`https://www.okx.com/api/v5/market/candles`
- Binance API：`https://api.binance.com/api/v3/klines`
- 请求超时：10秒
- 数据按时间戳升序排列（从旧到新）
- 自动处理不同交易所的数据格式差异
- OKX 返回 `datetime` 字段，Binance 返回 `timestamp` 字段

---

### 10. 加密货币页面

**GET** `/crypto.html`

返回加密货币交易数据页面，用于实时K线数据展示和分析。

**响应：** HTML 页面

---

## 使用流程

### 基本预测流程（文件数据）

1. **检查模型状态**：`GET /api/model-status`
2. **加载模型**：`POST /api/load-model`
3. **获取数据文件**：`GET /api/data-files`
4. **加载数据**：`POST /api/load-data`
5. **执行预测**：`POST /api/predict`

### 直接数据预测流程

1. **检查模型状态**：`GET /api/model-status`
2. **加载模型**：`POST /api/load-model`
3. **直接预测**：`POST /api/predictkline`（传入K线数据）

### 实时数据获取流程

1. **获取交易所K线数据**：`POST /api/exchange-kline`
2. **数据分析和可视化**：通过 `/crypto.html` 页面
3. **结合预测模型**：将实时数据用于模型预测

### 完整的实时预测流程

1. **检查模型状态**：`GET /api/model-status`
2. **加载模型**：`POST /api/load-model`
3. **获取实时K线数据**：`POST /api/exchange-kline`
4. **使用实时数据进行预测**：`POST /api/predictkline`（将获取的K线数据传入）

## 数据格式要求

### 输入数据格式

支持的文件格式：
- CSV 文件（`.csv`）
- Feather 文件（`.feather`）

必需的数据列：
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价

可选的数据列：
- `volume`: 交易量
- `amount`: 交易额
- `timestamps`/`timestamp`/`date`: 时间戳

### 预测结果格式

**重要更新：** 所有预测输出现在都包含完整的时间信息：

预测结果包含以下字段：
- `timestamp`: 预测时间点（秒级时间戳，Unix时间戳格式）
- `datetime`: 预测时间点（格式化时间字符串，格式："YYYY-MM-DD HH:MM:SS"）
- `open`: 预测开盘价
- `high`: 预测最高价
- `low`: 预测最低价
- `close`: 预测收盘价
- `volume`: 预测交易量（如果原数据包含）
- `amount`: 预测交易额

**实际数据格式（用于对比）：**

当使用 `/api/predict` 接口且提供历史数据对比时，`actual_data` 数组中的每个元素也包含相同的时间格式：
- `timestamp`: 实际时间点（秒级时间戳）
- `datetime`: 实际时间点（格式化时间字符串）
- `open`, `high`, `low`, `close`: 实际OHLC价格
- `volume`, `amount`: 实际交易量和交易额

**时间戳说明：**
- 秒级时间戳：标准Unix时间戳，便于程序处理和计算
- 格式化时间：人类可读的时间格式，便于显示和调试
- 两种格式确保了API的灵活性和易用性

## 错误处理

常见错误情况：

1. **模型未加载**：
   ```json
   {"error": "Kronos model not loaded, please load model first"}
   ```

2. **模型不可用**：
   ```json
   {
     "code": 500,
     "message": "Model not available",
     "data": null
   }
   ```

3. **不支持的模型**：
   ```json
   {
     "code": 400,
     "message": "Unsupported model: invalid_model",
     "data": null
   }
   ```

4. **数据文件格式错误**：
   ```json
   {
     "code": 400,
     "message": "Missing required columns: ['open', 'high', 'low', 'close']"
   }
   ```

5. **数据长度不足**：
   ```json
   {
     "code": 400,
     "message": "Insufficient data length, need at least 400 rows"
   }
   ```

6. **交易所API错误**：
   ```json
   {
     "code": 400,
     "message": "不支持的交易所: invalid_exchange"
   }
   ```

7. **交易所API请求失败**：
   ```json
   {
     "code": 500,
     "message": "请求失败，HTTP状态码: 404"
   }
   ```

8. **交易所API返回错误**：
   ```json
   {
     "code": 400,
     "message": "OKX API返回错误: Invalid symbol"
   }
   ```

9. **K线数据格式错误**：
   ```json
   {
     "code": 400,
     "message": "K-line data cannot be empty"
   }
   ```

10. **K线数据处理失败**：
    ```json
    {
      "code": 400,
      "message": "Failed to process K-line data: Invalid timestamp format"
    }
    ```

11. **K线数据长度不足**：
    ```json
    {
      "code": 400,
      "message": "Insufficient data length, need at least 400 rows, got 100 rows"
    }
    ```

12. **所有K线数据无效**：
    ```json
    {
      "code": 400,
      "message": "All K-line data contains invalid OHLC values"
    }
    ```

## 配置说明

### 服务器配置
- 主机：`0.0.0.0`
- 端口：`7070`
- 调试模式：启用

### 目录结构
- 数据目录：`./data/`
- 模型目录：`./model/`
- 预测结果目录：`./prediction_results/`
- 模板目录：`./templates/`

### 模型配置
模型文件从 Hugging Face 仓库加载：
- Kronos 模型：`NeoQuasar/Kronos-*`
- 分词器：`NeoQuasar/Kronos-Tokenizer-*`

### 全局变量
- `tokenizer`: 全局分词器实例
- `model`: 全局模型实例
- `predictor`: 全局预测器实例
- `MODEL_AVAILABLE`: 模型可用性标志

## 技术实现细节

### 数据预处理
1. **时间戳处理**：自动检测和转换多种时间戳格式
2. **数据类型转换**：确保OHLC数据为数值类型
3. **缺失值处理**：自动删除包含NaN的行
4. **数据验证**：检查必需列的存在性

### 预测算法
1. **模型加载**：支持多种Kronos模型规格
2. **上下文管理**：根据模型配置设置上下文长度
3. **采样控制**：支持温度和Top-p采样参数
4. **时间序列生成**：自动生成未来时间戳

### 图表生成
1. **Plotly集成**：生成交互式K线图
2. **多数据源支持**：历史数据、预测数据、实际数据对比
3. **时间连续性**：确保图表时间轴的连续性
4. **自适应布局**：根据数据量调整显示范围

### 结果保存
1. **自动保存**：预测结果自动保存为JSON格式
2. **详细分析**：包含连续性分析和误差统计
3. **时间戳标记**：使用时间戳命名结果文件
4. **元数据记录**：保存预测参数和输入数据摘要

## 注意事项

1. **数据质量**：确保输入数据完整且格式正确
2. **内存使用**：大数据集可能需要较多内存
3. **预测时间**：模型预测时间取决于数据量和模型大小
4. **设备支持**：支持 CPU 和 GPU 设备（如果可用）
5. **文件保存**：预测结果自动保存到 `prediction_results` 目录
6. **时间戳格式**：支持秒级时间戳和多种日期时间格式
7. **交易所限制**：注意交易所API的调用频率限制
8. **网络连接**：获取实时数据需要稳定的网络连接
9. **模型兼容性**：确保使用兼容的模型和分词器版本
10. **数据连续性**：预测结果的时间连续性依赖于输入数据的时间间隔