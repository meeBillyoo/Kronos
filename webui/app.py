import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import requests
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型管理模块
import app_model
import app_predict
import app_crypto_kline

app = Flask(__name__)
CORS(app)

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    # 调用app_model.py中的函数
    data_files = app_model.load_data_files()
    return jsonify({
            'code': 0,
            'message': 'Success',
            'data': data_files
        })

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
        
        # 调用app_model.py中的函数
        df, error = app_model.load_data_file(file_path)
        
        if error:
            return jsonify({'error': f'Failed to load data: {error}'}), 500
        
        # 构建完整的数据信息，包含前端需要的所有字段
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else None,
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else None,
            'timeframe': 'auto-detected',  # 可以根据时间间隔计算
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close']  # 预测用的列
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        result = app_predict.predict_from_file(file_path, lookback, pred_len, temperature, top_p, sample_count)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predictkline', methods=['POST'])
def predictkline():
    """Perform prediction with direct K-line data input"""
    try:
        data = request.get_json()
        kline_data = data.get('kline_data')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        # 修复：使用正确的变量名 kline_data
        result = app_predict.predict_from_kline(kline_data, lookback, pred_len, temperature, top_p, sample_count)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
@app.route('/api/load-model', methods=['POST'])
def load_model():
    """加载模型 - 调用app-model.py中的函数"""
    try:
        data = request.get_json()
        model_key = data.get('model', 'kronos-base')
        device = data.get('device', 'cpu')
        # 直接调用app-model.py中的load_model函数
        result = app_model.load_model(model_key, device)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'Model loading failed: {str(e)}',
            'data': None
        })

@app.route('/api/available-models')
def get_available_models():
    """获取可用模型列表 - 调用app-model.py中的函数"""
    try:
        # 直接调用app-model.py中的get_available_models函数
        result = app_model.get_available_models()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'Failed to get available models: {str(e)}',
            'data': None
        })

@app.route('/api/model-status')
def get_model_status():
    """获取模型状态 - 调用app-model.py中的函数"""
    try:
        # 直接调用app-model.py中的get_model_status函数
        result = app_model.get_model_status()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'Failed to get model status: {str(e)}',
            'data': None
        })

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/crypto.html')
def crypto_page():
    return render_template('crypto.html')

@app.route('/api/exchange-kline', methods=['POST'])
def exchange_kline_api():
    """获取交易所K线数据API接口"""
    try:
        # 获取请求参数
        data = request.get_json()
        # 参数提取和验证
        exchange = data.get('exchange', 'okx').lower().strip()
        symbol = data.get('symbol', 'BTC-USDT').strip()
        timeframe = data.get('timeframe', '1H').strip()
        limit = data.get('limit', 100)
        
        # 参数类型验证
        if not isinstance(limit, int):
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                return jsonify({
                    'code': 400,
                    'message': 'limit参数必须是整数',
                    'data': None
                }), 400
        
        # 参数范围验证
        if not all([exchange, symbol, timeframe]):
            return jsonify({
                'code': 400,
                'message': '参数不完整，请检查exchange、symbol、timeframe参数',
                'data': None
            }), 400
        
        if limit <= 0 or limit > 1000:
            return jsonify({
                'code': 400,
                'message': 'limit参数必须在1-1000之间',
                'data': None
            }), 400
        
        # 调用K线数据获取函数
        result, status_code = app_crypto_kline.get_exchange_kline(exchange, symbol, timeframe, limit)
        
        # 返回结果
        return result, status_code
        
    except Exception as e:
        return jsonify({
            'code': 500,
            'message': f'获取K线数据失败: {str(e)}',
            'data': None
        }), 500

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    if app_model:
        print("Model module available: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Model module not available: Will use simulated data for demonstration")
    
    app.run(debug=False, host='0.0.0.0', port=8068)
