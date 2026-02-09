"""
Telegram Bot Webhook Module for TVI Inference Results.

Usage:
1. Create .env file with:
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id

2. Import and call from inference.py:
   from autobot_webhook import send_inference_results
   send_inference_results(results_dict)
"""
import os
import requests
from typing import Dict, Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Load credentials from environment variables
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram_message(message: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """
    Send a message via Telegram Bot API.
    
    Args:
        message: The message to send (supports Markdown).
        token: Bot token (defaults to TELEGRAM_BOT_TOKEN env var).
        chat_id: Chat ID (defaults to TELEGRAM_CHAT_ID env var).
    
    Returns:
        True if successful, False otherwise.
    """
    token = token or BOT_TOKEN
    chat_id = chat_id or CHAT_ID
    
    if not token or not chat_id:
        print("⚠️ Telegram credentials not set. Skipping notification.")
        print("   Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram notification sent successfully!")
            return True
        else:
            print(f"❌ Telegram send failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False


def send_inference_results(results: Dict, dataset_name: str = "Unknown") -> bool:
    """
    Format and send inference results to Telegram.
    
    Args:
        results: Dictionary containing inference results.
        dataset_name: Name of the dataset (e.g., "CIFAR-100").
    
    Returns:
        True if successful, False otherwise.
    """
    message = f"🔬 *TVI Inference Complete*\n\n"
    message += f"📊 *Dataset*: `{dataset_name}`\n\n"
    
    # ID Metrics
    if "parametric_accuracy" in results:
        message += f"*ID Results:*\n"
        message += f"  • Param Acc: `{results.get('parametric_accuracy', 0):.2f}%`\n"
        message += f"  • Fused Acc: `{results.get('fused_accuracy', 0):.2f}%`\n"
        message += f"  • ECE: `{results.get('ece', 0):.4f}`\n\n"
    
    # OOD Metrics
    if "ood" in results:
        message += f"*OOD Results:*\n"
        for ood_name, ood_metrics in results["ood"].items():
            auroc = ood_metrics.get("auroc_fusion", 0)
            fpr = ood_metrics.get("fpr95_fusion", 0)
            message += f"  • {ood_name}: AUROC=`{auroc:.4f}` | FPR@95=`{fpr:.4f}`\n"
    
    message += f"\n⏰ _Report generated automatically_"
    
    return send_telegram_message(message)


# Test function
if __name__ == "__main__":
    # Test mode - check if credentials are set
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Credentials not set!")
        print("   Run: export TELEGRAM_BOT_TOKEN='your_token'")
        print("   Run: export TELEGRAM_CHAT_ID='your_chat_id'")
    else:
        test_message = "👋 TVI Webhook Test - Connection successful!"
        send_telegram_message(test_message)