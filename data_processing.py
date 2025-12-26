"""Data processing module for token selection and validation."""


def get_token_selection_mode():
    """Get user preference for token selection.
    
    Returns:
        tuple: (mode, manual_tokens) where mode is 'auto', 'manual', or 'hybrid'
    """
    print(f"\n" + "="*60)
    print("TOKEN SELECTION MODE")
    print("="*60)
    
    while True:
        print("\nHow would you like to select tokens for analysis?")
        print("1. Auto - Use top tokens by market cap (recommended for discovery)")
        print("2. Manual - Enter specific token symbols (up to 10 tokens)")
        print("3. Hybrid - Manual tokens + top performers to fill portfolio")
        print("4. View example token formats")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            return 'auto', []
        elif choice == '2':
            return 'manual', get_manual_token_list()
        elif choice == '3':
            return 'hybrid', get_manual_token_list()
        elif choice == '4':
            show_token_examples()
            continue
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def show_token_examples():
    """Show examples of token symbol formats."""
    print("\nTOKEN SYMBOL EXAMPLES:")
    print("• Bitcoin: BTC or BTCUSDT")
    print("• Ethereum: ETH or ETHUSDT") 
    print("• XRP: XRP or XRPUSDT")
    print("• Solana: SOL or SOLUSDT")
    print("• Cardano: ADA or ADAUSDT")
    print("• Binance Coin: BNB or BNBUSDT")
    print("• Polygon: MATIC or MATICUSDT")
    print("• Chainlink: LINK or LINKUSDT")
    print("• Polkadot: DOT or DOTUSDT")
    print("• Avalanche: AVAX or AVAXUSDT")
    print("\nNOTES:")
    print("• You can use either format (BTC or BTCUSDT)")
    print("• System will automatically add USDT if needed")
    print("• Maximum 10 tokens per analysis")
    print("• Minimum 2 tokens required for portfolio analysis")


def get_manual_token_list():
    """Get manual token list from user.
    
    Returns:
        list: Normalized list of token symbols
    """
    while True:
        tokens_input = input("\nEnter token symbols (comma-separated, max 10): ").strip()
        
        if not tokens_input:
            print("Please enter at least one token symbol.")
            continue
            
        # Parse and clean token list
        raw_tokens = [t.strip().upper() for t in tokens_input.split(',')]
        raw_tokens = [t for t in raw_tokens if t]  # Remove empty strings
        
        if len(raw_tokens) > 10:
            print(f"Too many tokens ({len(raw_tokens)}). Maximum is 10.")
            continue
            
        if len(raw_tokens) < 1:
            print("Please enter at least one token symbol.")
            continue
        
        # Normalize token symbols (add USDT if not present)
        normalized_tokens = []
        for token in raw_tokens:
            if token.endswith('USDT'):
                normalized_tokens.append(token)
            else:
                normalized_tokens.append(f"{token}USDT")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in normalized_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        print(f"\nNormalized tokens: {', '.join(unique_tokens)}")
        
        # Confirm selection
        confirm = input("Proceed with these tokens? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return unique_tokens
        else:
            print("Please re-enter your token selection.")


def validate_manual_tokens(tokens, price_data, market_data):
    """Validate that manual tokens have sufficient data.
    
    Args:
        tokens: List of token symbols to validate
        price_data: Dictionary of price DataFrames
        market_data: Dictionary of market data
        
    Returns:
        list or None: Valid tokens or None if insufficient valid tokens
    """
    print(f"\nValidating {len(tokens)} manually selected tokens...")
    
    valid_tokens = []
    invalid_tokens = []
    insufficient_data = []
    
    for token in tokens:
        if token not in price_data:
            invalid_tokens.append(token)
            continue
            
        # Check data quality
        df = price_data[token]
        if len(df) < 365:  # Need at least 1 year
            insufficient_data.append(f"{token} ({len(df)} days)")
            continue
            
        valid_tokens.append(token)
    
    # Report validation results
    if valid_tokens:
        print(f"✓ Valid tokens ({len(valid_tokens)}): {', '.join(valid_tokens)}")
    
    if invalid_tokens:
        print(f"✗ Invalid/unavailable tokens ({len(invalid_tokens)}): {', '.join(invalid_tokens)}")
        
    if insufficient_data:
        print(f"⚠ Insufficient data tokens: {', '.join(insufficient_data)}")
    
    if len(valid_tokens) < 2:
        print(f"\n❌ Error: Need at least 2 valid tokens for portfolio analysis.")
        print(f"   Only {len(valid_tokens)} valid tokens found.")
        return None
        
    return valid_tokens


def collect_tokens_by_mode(collector, mode, manual_tokens):
    """Collect token data based on selection mode.
    
    Args:
        collector: DataCollector instance
        mode: Selection mode ('auto', 'manual', or 'hybrid')
        manual_tokens: List of manually selected tokens (if applicable)
        
    Returns:
        tuple: (price_data, market_data) dictionaries
    """
    print(f"\nCollecting data for {mode} token selection...")
    
    if mode == 'auto':
        # Original behavior - get top tokens by market cap
        print("Using automatic token selection (top by market cap)")
        return collector.collect_all_data()
        
    elif mode == 'manual':
        # Only collect data for manually specified tokens
        print(f"Collecting data for {len(manual_tokens)} manually selected tokens")
        return collector.collect_all_data(symbols=manual_tokens)
        
    elif mode == 'hybrid':
        # Get manual tokens + top performers to fill gaps
        print(f"Using hybrid approach: {len(manual_tokens)} manual + top performers")
        
        # First get manual tokens
        manual_price_data, manual_market_data = collector.collect_all_data(symbols=manual_tokens)
        
        # If we have fewer than 10 total tokens, add top performers
        if len(manual_price_data) < 10:
            needed = 10 - len(manual_price_data)
            print(f"Adding {needed} top performers to reach 10 tokens...")
            
            # Get top tokens, excluding ones we already have
            all_price_data, all_market_data = collector.collect_all_data()
            
            # Combine datasets
            combined_price_data = manual_price_data.copy()
            combined_market_data = manual_market_data.copy()
            
            added = 0
            for token in all_price_data:
                if token not in combined_price_data and added < needed:
                    combined_price_data[token] = all_price_data[token]
                    if token in all_market_data:
                        combined_market_data[token] = all_market_data[token]
                    added += 1
            
            print(f"✓ Added {added} additional tokens")
            return combined_price_data, combined_market_data
        else:
            return manual_price_data, manual_market_data
