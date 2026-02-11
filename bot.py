"""
Telegram Bot — US RS+T Screener
═══════════════════════════════
Commands:
  /start            — welcome & instructions
  /now              — run screener immediately
  /status           — check bot health
  /YYMMDD           — back-test a specific date (e.g. /250211)
  /YYMMDD SYMBOL    — diagnose a symbol on that date (e.g. /250211 AAPL)

Scheduled:
  Daily at GMT+8 06:00 via Application.job_queue (built-in).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import date, datetime, time

import pytz
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from screener import (
    ScreenResult,
    calc_forward_performance,
    diagnose_symbol,
    format_diagnose_msg,
    format_forward_msg,
    format_screening_msg,
    run_screening,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Regex: /YYMMDD  or  /YYMMDD SYMBOL
_BACKTEST_RE = re.compile(r"^/(\d{6})(?:\s+(\S+))?$")


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def _parse_yymmdd(text: str) -> date | None:
    """Parse 'YYMMDD' → date, or None if invalid."""
    try:
        return datetime.strptime(text, "%y%m%d").date()
    except ValueError:
        return None


async def _send_text(context: ContextTypes.DEFAULT_TYPE,
                     chat_id: str, text: str) -> None:
    """Send a message, falling back to plain text if Markdown fails."""
    try:
        await context.bot.send_message(
            chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN_V2,
        )
    except Exception as md_err:
        logger.warning("MarkdownV2 send failed (%s), sending plain", md_err)
        plain = text.replace("*", "").replace("`", "").replace("\\", "")
        await context.bot.send_message(chat_id=chat_id, text=plain)


async def _send_file(context: ContextTypes.DEFAULT_TYPE,
                     chat_id: str, path: str, caption: str) -> None:
    """Send a document file."""
    if not path or not os.path.exists(path):
        return
    with open(path, "rb") as fh:
        await context.bot.send_document(
            chat_id=chat_id, document=fh,
            filename=os.path.basename(path), caption=caption,
        )


async def _run_in_executor(func, *args):
    """Run a blocking function in a thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


# ─────────────────────────────────────────────────────────
#  /now  &  Scheduled Run
# ─────────────────────────────────────────────────────────

async def _execute_now(context: ContextTypes.DEFAULT_TYPE,
                       chat_id: str) -> None:
    """Run today's screening and send results + TXT."""
    tz = pytz.timezone(config.TIMEZONE)
    date_str = datetime.now(tz).strftime("%Y-%m-%d")
    logger.info("Running live screening for %s …", date_str)

    try:
        sr: ScreenResult = await _run_in_executor(run_screening)
    except Exception as exc:
        logger.error("Screening failed: %s", exc, exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ Screener error ({date_str}):\n{str(exc)[:500]}",
        )
        return

    msg = format_screening_msg(sr.top, date_str)
    await _send_text(context, chat_id, msg)
    await _send_file(context, chat_id, sr.txt_path,
                     "📎 TradingView watchlist")
    logger.info("Live screening sent.")


async def _scheduled_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback for the daily scheduled job (via job_queue)."""
    chat_id = config.TELEGRAM_CHAT_ID
    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID not set — skipping scheduled run")
        return
    await _execute_now(context, chat_id)


# ─────────────────────────────────────────────────────────
#  Command Handlers
# ─────────────────────────────────────────────────────────

async def cmd_start(update: Update,
                    ctx: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    await update.message.reply_text(
        f"👋 US RS+T Screener Bot\n\n"
        f"Your Chat ID: {cid}\n\n"
        f"Commands:\n"
        f"  /now — 立即篩選\n"
        f"  /YYMMDD — 回測指定日期\n"
        f"       例: /250101\n"
        f"  /YYMMDD SYMBOL — 診斷個股\n"
        f"       例: /250101 AAPL\n"
        f"  /status — 系統狀態\n\n"
        f"⏰ 每日自動發送: 06:00 GMT+8",
    )


async def cmd_now(update: Update,
                  ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ 篩選中，請稍候（約 10-20 分鐘）…")
    await _execute_now(ctx, str(update.effective_chat.id))


async def cmd_status(update: Update,
                     ctx: ContextTypes.DEFAULT_TYPE) -> None:
    tz = pytz.timezone(config.TIMEZONE)
    now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    jobs = ctx.application.job_queue.jobs()
    await update.message.reply_text(
        f"✅ Bot running\n"
        f"🕐 {now_str}\n"
        f"📅 {len(jobs)} scheduled job(s)\n"
        f"⏰ Next auto-scan: 06:00 GMT+8",
    )


# ─────────────────────────────────────────────────────────
#  /YYMMDD  &  /YYMMDD SYMBOL
# ─────────────────────────────────────────────────────────

async def cmd_backtest(update: Update,
                       ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle both:
      /250211          → full back-test with forward performance
      /250211 AAPL     → diagnose single symbol
    """
    text = update.message.text.strip()
    m = _BACKTEST_RE.match(text)
    if not m:
        await update.message.reply_text(
            "❓ 格式: /YYMMDD 或 /YYMMDD SYMBOL")
        return

    target = _parse_yymmdd(m.group(1))
    symbol = m.group(2)  # None if not provided

    if target is None:
        await update.message.reply_text(
            "❓ 無效日期格式，請使用 YYMMDD（例: /250211）")
        return

    if target >= date.today():
        await update.message.reply_text("❓ 回測日期必須早於今天")
        return

    date_str = target.strftime("%Y-%m-%d")
    chat_id = str(update.effective_chat.id)

    if symbol:
        # ── Diagnose a specific symbol ────────────────────
        await update.message.reply_text(
            f"⏳ 診斷 {symbol.upper()} @ {date_str} …"
            f"（約 10-20 分鐘）"
        )
        try:
            sr = await _run_in_executor(run_screening, target)
            checks = diagnose_symbol(symbol.upper(), sr)
            msg = format_diagnose_msg(symbol.upper(), checks, date_str)
        except Exception as exc:
            logger.error("Diagnose failed: %s", exc, exc_info=True)
            await update.message.reply_text(
                f"⚠️ 診斷失敗: {str(exc)[:500]}")
            return

        await update.message.reply_text(msg)

    else:
        # ── Full back-test ────────────────────────────────
        await update.message.reply_text(
            f"⏳ 回測 {date_str} …（約 10-20 分鐘）"
        )
        try:
            sr = await _run_in_executor(run_screening, target)
        except Exception as exc:
            logger.error("Backtest screening failed: %s", exc,
                         exc_info=True)
            await update.message.reply_text(
                f"⚠️ 篩選失敗: {str(exc)[:500]}")
            return

        msg = format_screening_msg(sr.top, date_str)

        # Forward performance
        try:
            perf = calc_forward_performance(sr)
            fwd_msg = format_forward_msg(perf, sr)
            msg += "\n" + fwd_msg
        except Exception as exc:
            logger.error("Forward perf calc failed: %s", exc,
                         exc_info=True)
            msg += "\n\n⚠️ 前瞻表現計算失敗"

        await _send_text(ctx, chat_id, msg)
        await _send_file(ctx, chat_id, sr.txt_path,
                         f"📎 回測 {date_str}")


# ─────────────────────────────────────────────────────────
#  Application Lifecycle
# ─────────────────────────────────────────────────────────

async def post_init(application: Application) -> None:
    """
    Called after Application.initialize().
    The event loop is guaranteed running here, so registering
    jobs in job_queue is safe.
    """
    tz = pytz.timezone(config.TIMEZONE)
    target_time = time(hour=6, minute=0, tzinfo=tz)  # GMT+8 06:00

    application.job_queue.run_daily(
        _scheduled_job,
        time=target_time,
        name="daily_screening",
    )
    logger.info("Scheduled daily screening at 06:00 %s", config.TIMEZONE)


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Add it to .env or set as environment variable."
        )

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    # Fixed commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("now", cmd_now))
    app.add_handler(CommandHandler("status", cmd_status))

    # Dynamic /YYMMDD handler — registered after fixed commands
    # so /start, /now, /status take priority
    app.add_handler(MessageHandler(
        filters.Regex(_BACKTEST_RE), cmd_backtest,
    ))

    logger.info("Bot starting …")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
