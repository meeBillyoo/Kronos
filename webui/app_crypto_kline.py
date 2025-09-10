# -*- coding: utf-8 -*-
"""
K线数据获取模块
支持从OKX和Binance交易所获取K线数据
作者: Kronos Team
创建时间: 2024
"""

import json
import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple, Union


def get_exchange_okx_kline(exchange: str, symbol: str, timeframe: str, limit: int) -> Tuple[Union[Dict[str, Any], Any], int]:
    """
    获取OKX交易所K线数据
    
    Args:
        exchange (str): 交易所名称
        symbol (str): 交易对符号，如 'BTC-USDT'
        timeframe (str): 时间周期，如 '1H', '4H', '1D'
        limit (int): 获取数据条数
    
    Returns:
        Tuple[Union[Dict[str, Any], Any], int]: 返回响应数据和HTTP状态码
    """
    try:
        # OKX API调用配置
        url = 'https://www.okx.com/api/v5/market/candles'
        params = {
            'instId': symbol,  # 产品ID，如BTC-USDT
            'bar': timeframe,  # K线周期
            'limit': str(limit)  # 获取数量
        }
        
        # 发送HTTP请求获取K线数据
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # 检查API响应是否成功
            if result.get('code') == '0' and result.get('data'):
                # 转换OKX数据格式为标准化格式
                kline_data = []
                for item in result['data']:
                    # OKX返回格式: [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
                    timestamp_ms = int(item[0])  # OKX返回的是毫秒时间戳
                    timestamp_sec = timestamp_ms // 1000  # 转换为秒级时间戳
                    
                    # 构建标准化的K线数据格式
                    kline_data.append({
                        'time': timestamp_sec,  # 秒级时间戳
                        'datetime': datetime.datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d %H:%M:%S'),  # 可读时间格式
                        'open': float(item[1]),    # 开盘价
                        'high': float(item[2]),    # 最高价
                        'low': float(item[3]),     # 最低价
                        'close': float(item[4]),   # 收盘价
                        'volume': float(item[5]),  # 成交量
                        'amount': float(item[6]) if len(item) > 6 else float(item[5]) * float(item[4])  # 成交额
                    })
                
                # 按时间戳排序（从旧到新）
                kline_data.sort(key=lambda x: x['time'])
                
                # 构建成功响应数据
                response_data = {
                    'code': 0,
                    'message': 'K线数据获取成功',
                    'data': {
                        'success': True,
                        'exchange': exchange.upper(),
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'count': len(kline_data),
                        'kline_data': kline_data,
                        'data_info': {
                            'rows': len(kline_data),
                            'columns': ['time', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                            'price_range': {
                                'min': min(item['low'] for item in kline_data) if kline_data else 0,
                                'max': max(item['high'] for item in kline_data) if kline_data else 0
                            },
                            'time_range': {
                                'start': datetime.datetime.fromtimestamp(kline_data[0]['time']).isoformat() if kline_data else None,
                                'end': datetime.datetime.fromtimestamp(kline_data[-1]['time']).isoformat() if kline_data else None
                            }
                        }
                    }
                }
                
                # 检查是否在Flask应用上下文中，决定返回格式
                try:
                    from flask import has_app_context, jsonify
                    if has_app_context():
                        return jsonify(response_data), 200
                    else:
                        return response_data, 200
                except ImportError:
                    return response_data, 200
            else:
                # API返回错误
                error_data = {
                    'code': 400,
                    'message': f'OKX API返回错误: {result.get("msg", "未知错误")}',
                    'data': None
                }
                try:
                    from flask import has_app_context, jsonify
                    if has_app_context():
                        return jsonify(error_data), 400
                    else:
                        return error_data, 400
                except ImportError:
                    return error_data, 400
        else:
            # HTTP请求失败
            error_data = {
                'code': 500,
                'message': f'请求失败，HTTP状态码: {response.status_code}',
                'data': None
            }
            try:
                from flask import has_app_context, jsonify
                if has_app_context():
                    return jsonify(error_data), 500
                else:
                    return error_data, 500
            except ImportError:
                return error_data, 500
            
    except requests.exceptions.Timeout:
        # 请求超时异常处理
        error_data = {
            'code': 500,
            'message': '请求超时，请稍后重试',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500
    except requests.exceptions.RequestException as e:
        # 网络请求异常处理
        error_data = {
            'code': 500,
            'message': f'网络请求异常: {str(e)}',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500
    except Exception as e:
        # 其他异常处理
        error_data = {
            'code': 500,
            'message': f'获取K线数据失败: {str(e)}',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500


def get_exchange_binance_kline(exchange: str, symbol: str, timeframe: str, limit: int) -> Tuple[Union[Dict[str, Any], Any], int]:
    """
    获取Binance交易所K线数据
    
    Args:
        exchange (str): 交易所名称
        symbol (str): 交易对符号，如 'BTC-USDT'
        timeframe (str): 时间周期，如 '1h', '4h', '1d'
        limit (int): 获取数据条数
    
    Returns:
        Tuple[Union[Dict[str, Any], Any], int]: 返回响应数据和HTTP状态码
    """
    try:
        # Binance API调用配置
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol.replace('-', ''),  # Binance使用BTCUSDT格式，去掉中间的横线
            'interval': timeframe.lower(),      # 时间间隔，如1h, 4h, 1d
            'limit': limit                      # 获取数量
        }
        
        # 发送HTTP请求获取K线数据
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # 转换Binance数据格式为标准化格式
            kline_data = []
            for item in result:
                # Binance返回格式: [openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, ...]
                timestamp_ms = int(item[0])  # Binance返回的是毫秒时间戳
                timestamp_sec = timestamp_ms // 1000  # 转换为秒级时间戳
                
                # 构建标准化的K线数据格式
                kline_data.append({
                    'time': timestamp_sec,     # 秒级时间戳
                    'datetime': datetime.datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d %H:%M:%S'),  # 可读时间格式
                    'open': float(item[1]),    # 开盘价
                    'high': float(item[2]),    # 最高价
                    'low': float(item[3]),     # 最低价
                    'close': float(item[4]),   # 收盘价
                    'volume': float(item[5]),  # 成交量
                    'amount': float(item[7])   # Binance的成交额在索引7
                })
            
            # 按时间戳排序（从旧到新）
            kline_data.sort(key=lambda x: x['time'])
            
            # 构建成功响应数据
            response_data = {
                'code': 0,
                'message': 'K线数据获取成功',
                'data': {
                    'success': True,
                    'exchange': exchange.upper(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'count': len(kline_data),
                    'kline_data': kline_data,
                    'data_info': {
                        'rows': len(kline_data),
                        'columns': ['time', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                        'price_range': {
                            'min': min(item['low'] for item in kline_data) if kline_data else 0,
                            'max': max(item['high'] for item in kline_data) if kline_data else 0
                        },
                        'time_range': {
                            'start': datetime.datetime.fromtimestamp(kline_data[0]['time']).isoformat() if kline_data else None,
                            'end': datetime.datetime.fromtimestamp(kline_data[-1]['time']).isoformat() if kline_data else None
                        }
                    }
                }
            }
            
            # 检查是否在Flask应用上下文中，决定返回格式
            try:
                from flask import has_app_context, jsonify
                if has_app_context():
                    return jsonify(response_data), 200
                else:
                    return response_data, 200
            except ImportError:
                return response_data, 200
        else:
            # HTTP请求失败
            error_data = {
                'code': 500,
                'message': f'Binance API请求失败，HTTP状态码: {response.status_code}',
                'data': None
            }
            try:
                from flask import has_app_context, jsonify
                if has_app_context():
                    return jsonify(error_data), 500
                else:
                    return error_data, 500
            except ImportError:
                return error_data, 500
            
    except requests.exceptions.Timeout:
        # 请求超时异常处理
        error_data = {
            'code': 500,
            'message': '请求超时，请稍后重试',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500
    except requests.exceptions.RequestException as e:
        # 网络请求异常处理
        error_data = {
            'code': 500,
            'message': f'网络请求异常: {str(e)}',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500
    except Exception as e:
        # 其他异常处理
        error_data = {
            'code': 500,
            'message': f'获取K线数据失败: {str(e)}',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500


def get_exchange_kline(exchange: str, symbol: str, timeframe: str, limit: int) -> Tuple[Union[Dict[str, Any], Any], int]:
    """
    统一的交易所K线数据获取接口
    支持多个交易所的K线数据获取，提供统一的调用方式
    
    Args:
        exchange (str): 交易所名称 ('okx' 或 'binance')
        symbol (str): 交易对符号，如 'BTC-USDT'
        timeframe (str): 时间周期，如 '1H', '4H', '1D' (OKX) 或 '1h', '4h', '1d' (Binance)
        limit (int): 获取数据条数，范围1-1000
    
    Returns:
        Tuple[Union[Dict[str, Any], Any], int]: 返回响应数据和HTTP状态码
    """
    try:
        # 参数完整性验证
        if not all([exchange, symbol, timeframe]):
            error_data = {
                'code': 400,
                'message': '参数不完整，请检查exchange、symbol、timeframe参数',
                'data': None
            }
            try:
                from flask import has_app_context, jsonify
                if has_app_context():
                    return jsonify(error_data), 400
                else:
                    return error_data, 400
            except ImportError:
                return error_data, 400
        
        # 数据条数范围验证
        if limit <= 0 or limit > 1000:
            error_data = {
                'code': 400,
                'message': 'limit参数必须在1-1000之间',
                'data': None
            }
            try:
                from flask import has_app_context, jsonify
                if has_app_context():
                    return jsonify(error_data), 400
                else:
                    return error_data, 400
            except ImportError:
                return error_data, 400
        
        # 根据交易所类型调用相应的函数
        exchange_lower = exchange.lower()
        if exchange_lower == 'okx':
            return get_exchange_okx_kline(exchange, symbol, timeframe, limit)
        elif exchange_lower == 'binance':
            return get_exchange_binance_kline(exchange, symbol, timeframe, limit)
        else:
            error_data = {
                'code': 400,
                'message': f'不支持的交易所: {exchange}，目前支持: okx, binance',
                'data': None
            }
            try:
                from flask import has_app_context, jsonify
                if has_app_context():
                    return jsonify(error_data), 400
                else:
                    return error_data, 400
            except ImportError:
                return error_data, 400
            
    except Exception as e:
        # 统一异常处理
        error_data = {
            'code': 500,
            'message': f'获取K线数据失败: {str(e)}',
            'data': None
        }
        try:
            from flask import has_app_context, jsonify
            if has_app_context():
                return jsonify(error_data), 500
            else:
                return error_data, 500
        except ImportError:
            return error_data, 500


def run_kline_tests():
    """
    K线数据获取模块测试函数
    测试OKX和Binance交易所的K线数据获取功能
    包含功能测试、参数验证测试和错误处理测试
    """
    import time
    
    def test_exchange_kline():
        """测试交易所K线数据获取功能"""
        print("=" * 60)
        print("K线数据获取模块测试")
        print("=" * 60)
        
        # 测试用例配置
        test_cases = [
            {
                'name': 'OKX BTC-USDT 1小时K线',
                'exchange': 'okx',
                'symbol': 'BTC-USDT',
                'timeframe': '1H',
                'limit': 10
            },
            {
                'name': 'OKX ETH-USDT 4小时K线',
                'exchange': 'okx',
                'symbol': 'ETH-USDT',
                'timeframe': '4H',
                'limit': 5
            },
            {
                'name': 'Binance BTC-USDT 1小时K线',
                'exchange': 'binance',
                'symbol': 'BTC-USDT',
                'timeframe': '1h',
                'limit': 10
            },
            {
                'name': 'Binance ETH-USDT 4小时K线',
                'exchange': 'binance',
                'symbol': 'ETH-USDT',
                'timeframe': '4h',
                'limit': 5
            }
        ]
        
        success_count = 0
        total_count = len(test_cases)
        
        # 执行测试用例
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{total_count}] 测试: {test_case['name']}")
            print(f"参数: {test_case['exchange']} | {test_case['symbol']} | {test_case['timeframe']} | {test_case['limit']}条")
            
            try:
                # 调用K线数据获取函数
                result, status_code = get_exchange_kline(
                    exchange=test_case['exchange'],
                    symbol=test_case['symbol'],
                    timeframe=test_case['timeframe'],
                    limit=test_case['limit']
                )
                
                # 解析响应数据
                if hasattr(result, 'get_json'):
                    data = result.get_json()
                else:
                    data = result.json if hasattr(result, 'json') else result
                
                # 验证响应结果
                if status_code == 200 and data.get('code') == 0:
                    kline_data = data['data']['kline_data']
                    print(f"✅ 成功获取 {len(kline_data)} 条K线数据")
                    
                    if kline_data:
                        # 显示数据详情
                        first_item = kline_data[0]
                        last_item = kline_data[-1]
                        
                        print(f"   时间范围: {first_item['datetime']} ~ {last_item['datetime']}")
                        print(f"   价格范围: {data['data']['data_info']['price_range']['min']:.2f} ~ {data['data']['data_info']['price_range']['max']:.2f}")
                        print(f"   最新价格: 开盘={last_item['open']:.2f}, 收盘={last_item['close']:.2f}, 成交量={last_item['volume']:.2f}")
                    
                    success_count += 1
                else:
                    print(f"❌ 获取失败: {data.get('message', '未知错误')} (状态码: {status_code})")
                    
            except Exception as e:
                print(f"❌ 测试异常: {str(e)}")
            
            # 添加延迟避免API限制
            if i < total_count:
                print("   等待1秒...")
                time.sleep(1)
        
        # 测试结果统计
        print("\n" + "=" * 60)
        print(f"测试完成: {success_count}/{total_count} 个测试用例通过")
        if success_count == total_count:
            print("🎉 所有测试用例均通过！")
        else:
            print(f"⚠️  有 {total_count - success_count} 个测试用例失败")
        print("=" * 60)
    
    def test_parameter_validation():
        """测试参数验证功能"""
        print("\n📋 参数验证测试")
        print("-" * 40)
        
        # 无效参数测试用例
        invalid_cases = [
            {'exchange': '', 'symbol': 'BTC-USDT', 'timeframe': '1H', 'limit': 10, 'desc': '空交易所名称'},
            {'exchange': 'okx', 'symbol': '', 'timeframe': '1H', 'limit': 10, 'desc': '空交易对'},
            {'exchange': 'okx', 'symbol': 'BTC-USDT', 'timeframe': '', 'limit': 10, 'desc': '空时间周期'},
            {'exchange': 'okx', 'symbol': 'BTC-USDT', 'timeframe': '1H', 'limit': 0, 'desc': 'limit为0'},
            {'exchange': 'okx', 'symbol': 'BTC-USDT', 'timeframe': '1H', 'limit': 1001, 'desc': 'limit超过1000'},
            {'exchange': 'unknown', 'symbol': 'BTC-USDT', 'timeframe': '1H', 'limit': 10, 'desc': '不支持的交易所'}
        ]
        
        validation_success = 0
        
        for i, case in enumerate(invalid_cases, 1):
            print(f"[{i}] 测试: {case['desc']}")
            try:
                result, status_code = get_exchange_kline(
                    exchange=case['exchange'],
                    symbol=case['symbol'],
                    timeframe=case['timeframe'],
                    limit=case['limit']
                )
                
                # 解析响应数据
                if hasattr(result, 'get_json'):
                    data = result.get_json()
                else:
                    data = result.json if hasattr(result, 'json') else result
                
                # 验证是否正确返回错误
                if status_code != 200 and data.get('code') != 0:
                    print(f"   ✅ 正确识别无效参数: {data.get('message')}")
                    validation_success += 1
                else:
                    print(f"   ❌ 未能识别无效参数")
                    
            except Exception as e:
                print(f"   ❌ 验证异常: {str(e)}")
        
        print(f"\n参数验证测试完成: {validation_success}/{len(invalid_cases)} 个测试通过")
    
    def test_error_handling():
        """测试错误处理功能"""
        print("\n🔧 错误处理测试")
        print("-" * 40)
        
        # 测试网络错误处理（使用无效URL）
        print("[1] 测试网络错误处理")
        try:
            # 临时修改URL来模拟网络错误
            original_get = requests.get
            def mock_get(*args, **kwargs):
                raise requests.exceptions.RequestException("模拟网络错误")
            
            requests.get = mock_get
            result, status_code = get_exchange_kline('okx', 'BTC-USDT', '1H', 10)
            requests.get = original_get  # 恢复原始函数
            
            if hasattr(result, 'get_json'):
                data = result.get_json()
            else:
                data = result.json if hasattr(result, 'json') else result
            
            if status_code == 500 and '网络请求异常' in data.get('message', ''):
                print("   ✅ 网络错误处理正常")
            else:
                print("   ❌ 网络错误处理异常")
                
        except Exception as e:
            print(f"   ❌ 错误处理测试异常: {str(e)}")
        finally:
            requests.get = original_get  # 确保恢复原始函数
        
        print("错误处理测试完成")
    
    # 执行所有测试
    test_exchange_kline()      # 功能测试
    test_parameter_validation()  # 参数验证测试
    test_error_handling()      # 错误处理测试
    
    print("\n" + "=" * 60)
    print("🏁 所有测试执行完成")
    print("=" * 60)


if __name__ == '__main__':
    """
    模块主入口
    当直接运行此模块时，执行测试函数
    """
    print("启动K线数据获取模块测试...")
    run_kline_tests()
