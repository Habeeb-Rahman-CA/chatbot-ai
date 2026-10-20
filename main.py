#!/usr/bin/env python3
"""
Interactive Terminal Interface for Basic AI Chatbot.

Provides:
- Colorized command line chat
- Dynamic NLP debug inspection (/debug)
- State & memory inspection (/context)
- Built-in slash commands
"""

import sys
from chatbot import BasicAIChatbot

# ANSI color codes for terminal styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner() -> None:
    print(f"{CYAN}{BOLD}")
    print("=" * 60)
    print("       🤖 Basic AI Chatbot — NLP & AI Interaction       ")
    print("=" * 60)
    print(f"{RESET}{DIM}")
    print("Learn NLP in action:")
    print("  • Tokenization & Stemming")
    print("  • TF-IDF Vectorization & Cosine Similarity")
    print("  • Intent Classification & Entity Extraction")
    print("  • Multi-turn Dialogue State Tracking")
    print(f"\nType {YELLOW}/debug{RESET}{DIM} to toggle NLP inspection mode.")
    print(f"Type {YELLOW}/help{RESET}{DIM} for guide or {YELLOW}/quit{RESET}{DIM} to exit.")
    print(f"{CYAN}{'-' * 60}{RESET}\n")


def print_debug_info(result) -> None:
    print(f"{MAGENTA}{DIM}┌── [NLP Pipeline Inspection] ───────────────")
    print(f"│ Preprocessed Tokens : {result.tokens}")
    print(f"│ Extracted Entities  : {result.extracted_entities or 'None'}")
    print(f"│ Predicted Intent    : {result.intent}")
    print(f"│ Matched Pattern     : '{result.matched_pattern}'")
    print(f"│ Confidence Score    : {result.confidence * 100:.1f}%")
    print(f"│ Strategy            : {'Fallback' if result.is_fallback else 'Similarity Match'}")
    print(f"└─────────────────────────────────────────────{RESET}")


def main() -> None:
    bot = BasicAIChatbot()
    debug_mode = False

    print_banner()

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}PyBot: Goodbye! Have a great day!{RESET}")
            sys.exit(0)

        if not user_input:
            continue

        # Handle slash commands
        cmd = user_input.lower()
        if cmd in {"/quit", "/exit", "exit", "quit"}:
            print(f"{CYAN}PyBot: Goodbye! Have a great day!{RESET}")
            break

        if cmd == "/debug":
            debug_mode = not debug_mode
            status = f"{GREEN}ENABLED{RESET}" if debug_mode else f"{YELLOW}DISABLED{RESET}"
            print(f"{DIM}[System] NLP Debug mode is now {status}{RESET}\n")
            continue

        if cmd == "/context":
            ctx = bot.context
            print(f"{BLUE}┌── [Dialogue Context State] ─────────────────")
            print(f"│ User Name     : {ctx.user_name or '(not set yet)'}")
            print(f"│ Turn Count    : {ctx.turn_count}")
            print(f"│ Last Intent   : {ctx.last_intent}")
            print(f"│ History Length: {len(ctx.history)} messages")
            print(f"└─────────────────────────────────────────────{RESET}\n")
            continue

        if cmd == "/reset":
            bot.reset_conversation()
            print(f"{YELLOW}[System] Conversation memory reset successfully.{RESET}\n")
            continue

        if cmd == "/help":
            print(f"{CYAN}Available Commands:{RESET}")
            print("  /debug    - Toggle live NLP metadata (tokens, similarity, intent)")
            print("  /context  - View current conversation state & stored entities")
            print("  /reset    - Clear conversation memory")
            print("  /quit     - Exit chat")
            print(f"\n{CYAN}Sample inputs to try:{RESET}")
            print("  • 'Hello there!'")
            print("  • 'My name is Maya'")
            print("  • 'What is my name?'")
            print("  • 'What is NLP and how does it work?'")
            print("  • 'What time is it?'")
            print("  • 'Tell me a joke'")
            print("  • 'I am feeling wonderful today!'\n")
            continue

        # Process input through chatbot NLP pipeline
        result = bot.respond(user_input)

        if debug_mode:
            print_debug_info(result)

        print(f"{CYAN}{BOLD}PyBot > {RESET}{result.response}\n")


if __name__ == "__main__":
    main()

