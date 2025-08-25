# tools/trading_tool.py
import math
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json

from config import get_openai_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class QuestionAnalysis(BaseModel):
    """Kết quả phân tích câu hỏi từ LLM"""
    analysis_type: str = Field(description="Loại phân tích: 'overview', 'recent_trades', 'time_period', 'specific_metrics'")
    recent_n_trades: Optional[int] = Field(description="Số giao dịch gần đây nếu user hỏi 'n giao dịch gần đây'")
    time_period: Optional[Dict[str, Any]] = Field(description="Khoảng thời gian cụ thể nếu user hỏi về thời gian")
    specific_metrics: List[str] = Field(description="Các chỉ số cụ thể user muốn biết")
    focus_areas: List[str] = Field(description="Các lĩnh vực tập trung: 'performance', 'risk', 'timing', 'behavior'")

def analyze_user_question_with_llm(question: str) -> Dict[str, Any]:
    """
    Sử dụng LLM để phân tích câu hỏi và trả về các key cụ thể
    
    Args:
        question: Câu hỏi của user
        
    Returns:
        Dictionary với các key được phân tích
    """
    try:
        llm = get_openai_llm()
        parser = JsonOutputParser(pydantic_object=QuestionAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia phân tích câu hỏi trading. Nhiệm vụ của bạn là phân tích câu hỏi và trả về các thông tin cụ thể.

## Nhiệm vụ:
1. **Phân tích loại câu hỏi**: overview, recent_trades, time_period, specific_metrics
2. **Xác định số giao dịch gần đây**: nếu user hỏi "n giao dịch gần đây" hoặc tương tự
3. **Xác định khoảng thời gian**: nếu user hỏi về thời gian cụ thể
4. **Xác định chỉ số cụ thể**: các metrics user muốn biết
5. **Xác định lĩnh vực tập trung**: performance, risk, timing, behavior

## Ví dụ cụ thể:
- "Tình hình trading hiện tại của tôi thế nào?" → analysis_type: "overview", focus_areas: ["performance", "risk"]
- "Thời gian gần đây tôi trading có ổn không?" → analysis_type: "time_period", time_period: {{"period": "recent", "value": "last_30_days"}}, focus_areas: ["performance", "risk"]
- "Dựa vào kết quả trading, tôi phù hợp với phong cách gì?" → analysis_type: "overview", focus_areas: ["behavior", "performance", "risk"]
- "Kiểm tra 200 lệnh gần nhất của tôi và đưa ra phân tích" → analysis_type: "recent_trades", recent_n_trades: 200, focus_areas: ["performance", "risk", "timing"]

## Quy tắc:
- Nếu không có thông tin cụ thể, để null
- Luôn trả về JSON hợp lệ
- Phân tích chính xác ý định của user
- Ưu tiên hiểu ý nghĩa thực sự của câu hỏi"""),
            ("human", "Câu hỏi: {question}")
        ])
        
        chain = prompt | llm | parser
        
        result = chain.invoke({"question": question})
        print(f"🔍 LLM Analysis Result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        # Fallback to basic analysis
        return {
            "analysis_type": "overview",
            "recent_n_trades": None,
            "time_period": None,
            "specific_metrics": [],
            "focus_areas": ["performance", "risk"]
        }

def read_trading_excel(file_path: str) -> pd.DataFrame:
    """
    Read trading history from Excel file and return a cleaned DataFrame.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Cleaned DataFrame with standardized column names
    """
    try:
        # Read Excel file
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.xls'):
            df = pd.read_excel(file_path, engine='xlrd')
        else:
            raise ValueError("File must be .xlsx or .xls format")
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Standardize column names to match expected format
        column_mapping = {
            'symbol': ['symbol', 'cặp tiền', 'cặp', 'pair', 'instrument'],
            'side': ['side', ' side', 'hướng', 'direction', 'loại', 'type'],
            'close_time': ['close_time', ' close_time', 'thời gian đóng', 'ngày đóng', 'date', 'time'],
            'net_profit': ['net_profit', 'net_profit', 'lợi nhuận ròng', 'pnl', 'profit', 'lãi lỗ'],
            'profit_gross': ['profit_gross', 'profit_gross', 'lợi nhuận gộp', 'gross profit', 'lãi gộp'],
            'commission': ['commission', 'commission', 'phí giao dịch', 'phí', 'fee'],
            'swap': ['swap', 'swap', 'phí qua đêm', 'phí swap', 'overnight'],
            'balance_after': ['balance_after', 'balance_after', 'số dư sau', 'balance', 'số dư'],
            'pips': ['pips', 'pips', 'điểm', 'pip'],
            'volume_lots_closed': ['volume_lots_closed', 'volume_lots_closed', 'khối lượng đóng', 'volume', 'lot'],
            'quantity_closed': ['quantity_closed', 'quantity_closed', 'số lượng đóng', 'quantity', 'số lượng'],
            'open_price': ['open_price', ' open_price', 'giá mở', 'giá vào lệnh', 'entry price'],
            'close_price': ['close_price', ' close_price', 'giá đóng', 'giá thoát lệnh', 'exit price']
        }
        
        # Apply column mapping
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    if standard_name not in df.columns:
                        df[standard_name] = df[possible_name]
                    break
        
        # Validate required columns
        required_columns = ['symbol', 'side', 'close_time', 'net_profit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        raise

def get_trading_data_from_excel(file_path: str = None) -> pd.DataFrame:
    """
    Get trading data from Excel file. If no file path provided, look for common locations.
    
    Args:
        file_path: Optional path to Excel file
        
    Returns:
        DataFrame with trading data
    """
    if file_path and os.path.exists(file_path):
        return read_trading_excel(file_path)
    
    # Look for common locations
    common_paths = [
        "trading_data/test_sample.xlsx",
        "data/trading_history.xlsx",
        "trading_history.xlsx",
        "trading_data.xlsx"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"📁 Found trading data at: {path}")
            return read_trading_excel(path)
    
    # If no file found, create sample data for demonstration
    print("⚠️ No trading data file found. Using sample data for demonstration.")
    return pd.DataFrame([
        {"symbol": "XAUUSD", "side": "BUY", "close_time": "01/01/2024 10:00", "net_profit": 50, "commission": -2, "swap": -1},
        {"symbol": "XAUUSD", "side": "SELL", "close_time": "01/01/2024 11:00", "net_profit": -30, "commission": -2, "swap": -1},
    ])

def calculate_trade_index(df: pd.DataFrame):
    """
    Calculate trade index and analytics from user's trade history.
    Returns comprehensive summary of all possible metrics.
    """
    # Normalize column names
    if hasattr(df, 'columns'):
        df = df.rename(columns={c: c.strip() for c in df.columns})

    # Map Vietnamese side values to English
    if 'side' in df.columns:
        df['side'] = df['side'].replace({'Mua': 'BUY', 'Bán': 'SELL'})

    # Parse close_time to datetime
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce', dayfirst=True)
        df['hour'] = df['close_time'].dt.hour
        df['date'] = df['close_time'].dt.date

    # Coerce numeric fields
    numeric_cols = [
        'commission', 'swap', 'profit_gross', 'net_profit', 'balance_after',
        'pips', 'volume_lots_closed', 'quantity_closed', 'open_price', 'close_price'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Determine volume column
    volume_col = 'volume_lots_closed' if 'volume_lots_closed' in df.columns else (
        'quantity_closed' if 'quantity_closed' in df.columns else None
    )

    # Basic metrics
    trades = int(len(df))
    total_commission = float(df['commission'].sum()) if 'commission' in df.columns else 0.0
    total_swap = float(df['swap'].sum()) if 'swap' in df.columns else 0.0
    total_fees = abs(total_commission) + abs(total_swap)

    # Win/Loss analysis
    win_trades_df = df[df['net_profit'] > 0]
    loss_trades_df = df[df['net_profit'] < 0]
    wins_sum = float(win_trades_df['net_profit'].sum())
    losses_sum = float(loss_trades_df['net_profit'].sum())

    win_rate_pct = (len(win_trades_df) / trades * 100.0) if trades > 0 else 0.0
    net_profit = wins_sum + losses_sum
    avg_profit_per_trade = net_profit / trades if trades > 0 else 0.0
    
    # Performance metrics
    avg_profit_win = wins_sum / len(win_trades_df) if len(win_trades_df) > 0 else 0.0
    avg_loss_loss = losses_sum / len(loss_trades_df) if len(loss_trades_df) > 0 else 0.0
    profit_factor = abs(wins_sum / losses_sum) if losses_sum != 0 else float('inf')
    
    # Best/Worst trades
    best_trade = float(df['net_profit'].max()) if 'net_profit' in df.columns else 0.0
    worst_trade = float(df['net_profit'].min()) if 'net_profit' in df.columns else 0.0
    
    # Risk metrics
    max_consecutive_losses = 0
    current_streak = 0
    for profit in df['net_profit']:
        if profit < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0
    
    # Drawdown calculation
    if 'balance_after' in df.columns:
        balance_series = df['balance_after'].dropna()
        if len(balance_series) > 0:
            running_max = balance_series.expanding().max()
            drawdown = (balance_series - running_max) / running_max * 100
            max_drawdown_pct = float(drawdown.min())
        else:
            max_drawdown_pct = 0.0
    else:
        max_drawdown_pct = 0.0
    
    # Time analysis
    time_analysis = {}
    if 'hour' in df.columns:
        for hour in range(24):
            hour_trades = df[df['hour'] == hour]
            if len(hour_trades) > 0:
                time_analysis[hour] = {
                    'trades': len(hour_trades),
                    'profit': float(hour_trades['net_profit'].sum()),
                    'win_rate': len(hour_trades[hour_trades['net_profit'] > 0]) / len(hour_trades) * 100
                }
    
    # Symbol analysis
    symbol_analysis = {}
    if 'symbol' in df.columns:
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol]
            symbol_analysis[symbol] = {
                'trades': len(symbol_trades),
                'profit': float(symbol_trades['net_profit'].sum()),
                'win_rate': len(symbol_trades[symbol_trades['net_profit'] > 0]) / len(symbol_trades) * 100
            }
    
    # Side analysis
    side_analysis = {}
    if 'side' in df.columns:
        for side in df['side'].unique():
            side_trades = df[df['side'] == side]
            side_analysis[side] = {
                'trades': len(side_trades),
                'profit': float(side_trades['net_profit'].sum()),
                'win_rate': len(side_trades[side_trades['net_profit'] > 0]) / len(side_trades) * 100
            }
    
    # Behavioral metrics
    behavioral = {}
    if 'close_time' in df.columns:
        # Rapid fire detection (trades within 1 hour)
        df_sorted = df.sort_values('close_time')
        rapid_fire_count = 0
        for i in range(1, len(df_sorted)):
            time_diff = df_sorted.iloc[i]['close_time'] - df_sorted.iloc[i-1]['close_time']
            if time_diff.total_seconds() < 3600:  # 1 hour
                rapid_fire_count += 1
        
        behavioral['rapid_fire_ratio'] = rapid_fire_count / max(trades - 1, 1)
        behavioral['avg_time_between_trades'] = "N/A"  # Could be calculated if needed
    
    # Risk KPIs
    risk_kpi = {
        'max_risk_per_trade': abs(worst_trade),
        'limit_daily_stop': abs(worst_trade) * 3,  # Example: 3x worst trade
        'avgTradesPerDay': trades / max(len(df['date'].unique()), 1) if 'date' in df.columns else 0
    }
    
    # Pips analysis
    total_pips = 0
    if 'pips' in df.columns:
        total_pips = float(df['pips'].sum())
    
    # Volume analysis
    total_volume = 0
    if volume_col and volume_col in df.columns:
        total_volume = float(df[volume_col].sum())
    
    return {
        'trades': trades,
        'net_profit': net_profit,
        'win_rate_pct': win_rate_pct,
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_profit_win': avg_profit_win,
        'avg_loss_loss': avg_loss_loss,
        'profit_factor': profit_factor,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'max_consecutive_losses': max_consecutive_losses,
        'max_drawdown_pct': max_drawdown_pct,
        'total_commission': total_commission,
        'total_swap': total_swap,
        'total_fees': total_fees,
        'total_pips': total_pips,
        'total_volume': total_volume,
        'time_analysis': time_analysis,
        'symbol_analysis': symbol_analysis,
        'side_analysis': side_analysis,
        'behavioral': behavioral,
        'risk_kpi': risk_kpi,
        'raw_data': df
    }

def filter_trades_by_conditions(df: pd.DataFrame, analysis_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Lọc giao dịch theo điều kiện từ kết quả phân tích LLM
    
    Args:
        df: DataFrame gốc
        analysis_result: Kết quả phân tích từ LLM
        
    Returns:
        DataFrame đã lọc
    """
    filtered_df = df.copy()
    
    # Lọc theo số giao dịch gần đây
    if analysis_result.get('recent_n_trades'):
        n_trades = analysis_result['recent_n_trades']
        print(f"🔍 Lọc {n_trades} giao dịch gần đây")
        if 'close_time' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('close_time', ascending=False).head(n_trades)
            print(f"✅ Đã lọc thành công: {len(filtered_df)} giao dịch")
        else:
            # Nếu không có cột thời gian, lấy n dòng cuối
            filtered_df = filtered_df.tail(n_trades)
            print(f"⚠️ Không có cột thời gian, lấy {len(filtered_df)} dòng cuối")
    
    # Lọc theo khoảng thời gian
    if analysis_result.get('time_period'):
        time_period = analysis_result['time_period']
        print(f"🔍 Lọc theo thời gian: {time_period}")
        
        if 'close_time' in filtered_df.columns:
            current_time = datetime.now()
            
            if time_period.get('period') == 'recent' and time_period.get('value') == 'last_30_days':
                # 30 ngày gần đây
                start_date = current_time - timedelta(days=30)
                filtered_df = filtered_df[filtered_df['close_time'] >= start_date]
                print(f"✅ Đã lọc 30 ngày gần đây: {len(filtered_df)} giao dịch")
                
            elif time_period.get('period') == 'month':
                if time_period.get('value') == 'current':
                    # Tháng hiện tại
                    start_date = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                else:
                    # Tháng cụ thể
                    month_value = time_period.get('value', 1)
                    start_date = current_time.replace(month=month_value, day=1, hour=0, minute=0, second=0, microsecond=0)
                
                end_date = start_date + timedelta(days=32)
                end_date = end_date.replace(day=1) - timedelta(days=1)
                
                filtered_df = filtered_df[
                    (filtered_df['close_time'] >= start_date) & 
                    (filtered_df['close_time'] <= end_date)
                ]
                print(f"✅ Đã lọc theo tháng: {len(filtered_df)} giao dịch")
            
            elif time_period.get('period') == 'week':
                if time_period.get('value') == 'current':
                    # Tuần hiện tại
                    start_date = current_time - timedelta(days=current_time.weekday())
                    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    # Tuần cụ thể
                    week_value = time_period.get('value', 0)
                    start_date = current_time - timedelta(days=current_time.weekday() + week_value * 7)
                    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                
                end_date = start_date + timedelta(days=7)
                filtered_df = filtered_df[
                    (filtered_df['close_time'] >= start_date) & 
                    (filtered_df['close_time'] < end_date)
                ]
                print(f"✅ Đã lọc theo tuần: {len(filtered_df)} giao dịch")
        else:
            print("⚠️ Không có cột thời gian để lọc")
    
    return filtered_df

def generate_comprehensive_report(df: pd.DataFrame, analysis_result: Dict[str, Any] = None) -> str:
    """
    Tạo báo cáo toàn diện dựa trên dữ liệu giao dịch
    
    Args:
        df: DataFrame giao dịch
        analysis_result: Kết quả phân tích LLM (optional)
        
    Returns:
        Báo cáo toàn diện
    """
    # Lọc dữ liệu theo điều kiện nếu có
    if analysis_result:
        df = filter_trades_by_conditions(df, analysis_result)
    
    # Tính toán chỉ số
    result = calculate_trade_index(df)
    
    # Tạo báo cáo
    report_parts = []
    
    # Header
    if analysis_result and analysis_result.get('analysis_type') != 'overview':
        report_parts.append(f"📊 BÁO CÁO PHÂN TÍCH: {analysis_result['analysis_type'].upper()}")
        if analysis_result.get('recent_n_trades'):
            report_parts.append(f"🔍 Dựa trên {analysis_result['recent_n_trades']} giao dịch gần đây")
        if analysis_result.get('time_period'):
            time_info = analysis_result['time_period']
            report_parts.append(f"⏰ Khoảng thời gian: {time_info.get('period', 'N/A')} - {time_info.get('value', 'N/A')}")
    else:
        report_parts.append("📊 BÁO CÁO TỔNG QUAN GIAO DỊCH")
    
    report_parts.append("=" * 50)
    
    # Tổng quan
    report_parts.append("📈 TỔNG QUAN")
    report_parts.append(f"• Tổng số lệnh: {result['trades']}")
    report_parts.append(f"• Tổng lợi nhuận: {result['net_profit']:,.2f}")
    report_parts.append(f"• Tỷ lệ thắng: {result['win_rate_pct']:.1f}%")
    report_parts.append(f"• Lợi nhuận trung bình/lệnh: {result['avg_profit_per_trade']:,.2f}")
    
    # Hiệu suất
    report_parts.append("\n🎯 HIỆU SUẤT")
    report_parts.append(f"• Lệnh thắng trung bình: {result['avg_profit_win']:,.2f}")
    report_parts.append(f"• Lệnh thua trung bình: {result['avg_loss_loss']:,.2f}")
    report_parts.append(f"• Hệ số lợi nhuận: {result['profit_factor']:.2f}")
    report_parts.append(f"• Lệnh thắng nhất: {result['best_trade']:,.2f}")
    report_parts.append(f"• Lệnh thua nhất: {result['worst_trade']:,.2f}")
    
    # Rủi ro
    report_parts.append("\n⚠️ RỦI RO")
    report_parts.append(f"• Sụt giảm tối đa: {result['max_drawdown_pct']:.1f}%")
    report_parts.append(f"• Số lệnh thua liên tiếp tối đa: {result['max_consecutive_losses']}")
    report_parts.append(f"• Giới hạn rủi ro/lệnh: {result['risk_kpi']['max_risk_per_trade']:,.2f}")
    report_parts.append(f"• Trung bình giao dịch/ngày: {result['risk_kpi']['avgTradesPerDay']:.1f}")
    
    # Phân tích thời gian
    if result['time_analysis']:
        report_parts.append("\n⏰ PHÂN TÍCH THỜI GIAN")
        best_hour = max(result['time_analysis'].items(), key=lambda x: x[1]['profit'])
        worst_hour = min(result['time_analysis'].items(), key=lambda x: x[1]['profit'])
        report_parts.append(f"• Giờ tốt nhất: {best_hour[0]}:00 (lợi nhuận: {best_hour[1]['profit']:,.2f})")
        report_parts.append(f"• Giờ kém nhất: {worst_hour[0]}:00 (lợi nhuận: {worst_hour[1]['profit']:,.2f})")
    
    # Phân tích cặp tiền
    if result['symbol_analysis']:
        report_parts.append("\n💱 PHÂN TÍCH CẶP TIỀN")
        for symbol, data in result['symbol_analysis'].items():
            report_parts.append(f"• {symbol}: {data['trades']} lệnh, lợi nhuận: {data['profit']:,.2f}, win rate: {data['win_rate']:.1f}%")
    
    # Phân tích hướng giao dịch
    if result['side_analysis']:
        report_parts.append("\n🔄 PHÂN TÍCH HƯỚNG GIAO DỊCH")
        for side, data in result['side_analysis'].items():
            report_parts.append(f"• {side}: {data['trades']} lệnh, lợi nhuận: {data['profit']:,.2f}, win rate: {data['win_rate']:.1f}%")
    
    # Hành vi giao dịch
    if result['behavioral']:
        report_parts.append("\n🧠 PHÂN TÍCH HÀNH VI")
        report_parts.append(f"• Tỷ lệ giao dịch nhanh: {result['behavioral']['rapid_fire_ratio']:.1%}")
    
    # Chỉ số khác
    report_parts.append("\n📊 CHỈ SỐ KHÁC")
    report_parts.append(f"• Tổng phí giao dịch: {result['total_fees']:,.2f}")
    report_parts.append(f"• Tổng pips: {result['total_pips']:,.0f}")
    if result['total_volume'] > 0:
        report_parts.append(f"• Tổng khối lượng: {result['total_volume']:,.2f}")
    
    # Thêm đánh giá tổng quan dựa trên focus areas
    if analysis_result and analysis_result.get('focus_areas'):
        report_parts.append("\n🎯 ĐÁNH GIÁ TỔNG QUAN")
        focus_areas = analysis_result['focus_areas']
        
        if 'performance' in focus_areas:
            if result['win_rate_pct'] >= 60:
                report_parts.append("• Hiệu suất: 🟢 Tuyệt vời (>60% win rate)")
            elif result['win_rate_pct'] >= 50:
                report_parts.append("• Hiệu suất: 🟡 Tốt (50-60% win rate)")
            else:
                report_parts.append("• Hiệu suất: 🔴 Cần cải thiện (<50% win rate)")
        
        if 'risk' in focus_areas:
            if result['max_drawdown_pct'] <= 10:
                report_parts.append("• Quản lý rủi ro: 🟢 Tuyệt vời (≤10% drawdown)")
            elif result['max_drawdown_pct'] <= 20:
                report_parts.append("• Quản lý rủi ro: 🟡 Tốt (10-20% drawdown)")
            else:
                report_parts.append("• Quản lý rủi ro: 🔴 Cần cải thiện (>20% drawdown)")
        
        if 'behavior' in focus_areas:
            if result['behavioral'].get('rapid_fire_ratio', 0) <= 0.1:
                report_parts.append("• Hành vi giao dịch: 🟢 Kiểm soát tốt (ít giao dịch nhanh)")
            elif result['behavioral'].get('rapid_fire_ratio', 0) <= 0.3:
                report_parts.append("• Hành vi giao dịch: 🟡 Trung bình (một số giao dịch nhanh)")
            else:
                report_parts.append("• Hành vi giao dịch: 🔴 Cần cải thiện (nhiều giao dịch nhanh)")
    
    return "\n".join(report_parts)

def analyze_trading_data(file_path: str, question: str = None) -> Dict[str, Any]:
    """
    Phân tích dữ liệu giao dịch với câu hỏi cụ thể
    
    Args:
        file_path: Đường dẫn file Excel
        question: Câu hỏi của user (optional)
        
    Returns:
        Dictionary chứa kết quả phân tích và báo cáo
    """
    try:
        print(f"🔍 Bắt đầu phân tích dữ liệu trading...")
        print(f"📁 File: {file_path}")
        print(f"❓ Câu hỏi: {question or 'Không có'}")
        
        # Đọc dữ liệu
        df = get_trading_data_from_excel(file_path)
        print(f"✅ Đọc thành công: {len(df)} dòng dữ liệu")
        
        # Phân tích câu hỏi nếu có
        analysis_result = None
        if question:
            print(f"🧠 Phân tích câu hỏi với LLM...")
            analysis_result = analyze_user_question_with_llm(question)
            print(f"✅ Phân tích LLM hoàn thành")
        
        # Tạo báo cáo
        print(f"📊 Tạo báo cáo...")
        report = generate_comprehensive_report(df, analysis_result)
        print(f"✅ Báo cáo hoàn thành")
        
        # Tính toán chỉ số toàn bộ
        print(f"📈 Tính toán chỉ số toàn bộ...")
        full_result = calculate_trade_index(df)
        print(f"✅ Tính toán hoàn thành")
        
        result = {
            "success": True,
            "data": df,
            "analysis_result": analysis_result,
            "full_result": full_result,
            "report": report,
            "file_path": file_path,
            "question": question,
            "summary": {
                "total_trades": len(df),
                "analysis_type": analysis_result.get('analysis_type', 'overview') if analysis_result else 'overview',
                "report_length": len(report)
            }
        }
        
        print(f"🎉 Phân tích hoàn thành thành công!")
        return result
        
    except Exception as e:
        error_msg = f"Lỗi phân tích: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "file_path": file_path,
            "question": question
        }