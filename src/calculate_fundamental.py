from typing import Dict, Any

def calculate_fundamental_price(
    num_rounds: int,
    expected_dividend: float,
    interest_rate: float,
    redemption_value: float,
    current_period: int = 1
) -> float:
    """
    Calculate the fundamental price as described in the experiment:
    FVt = E(d) * sum((1+r)^(-τ) for τ from 1 to T-t+1) + K(1+r)^(-(T-t+1))
    
    Args:
        num_rounds: Total number of trading periods (T)
        expected_dividend: Expected dividend payment each period (E(d))
        interest_rate: Interest rate (r)
        redemption_value: Redemption value at end of final period (K)
        current_period: Current period (t), default is period 1
    """
    fundamental_price = 0.0
    
    # Number of remaining periods including the current one
    remaining_periods = num_rounds - current_period + 1
    
    # Calculate present value of expected dividends
    for t in range(1, remaining_periods + 1):
        discount_factor = 1 / ((1 + interest_rate) ** t)
        fundamental_price += expected_dividend * discount_factor
    
    # Add present value of redemption value at the end of trading period
    redemption_discount_factor = 1 / ((1 + interest_rate) ** remaining_periods)
    fundamental_price += redemption_value * redemption_discount_factor
    
    return fundamental_price

def calibrate_redemption_value(
    num_rounds: int,
    expected_dividend: float,
    interest_rate: float,
    target_fundamental: float
) -> float:
    """
    Calculate the redemption value needed to achieve a constant fundamental value.
    
    In the experiment, the redemption value is calibrated so that:
    1. The fundamental value is constant across periods
    2. The expected dividend return equals the interest rate
    """
    # For constant fundamental value, the redemption value K should satisfy:
    # target_fundamental = E(d)/r (for infinite horizon)
    # This is simplified from the standard Gordon growth model with no growth
    
    # Calculate the PV of dividends for all periods
    dividend_component = 0.0
    for t in range(1, num_rounds + 1):
        discount_factor = 1 / ((1 + interest_rate) ** t)
        dividend_component += expected_dividend * discount_factor
    
    # Calculate required redemption value
    required_present_value = target_fundamental - dividend_component
    required_redemption = required_present_value * ((1 + interest_rate) ** num_rounds)
    
    return required_redemption

def main():
    print("FUNDAMENTAL PRICE CALCULATION FOR EXPERIMENTAL DESIGN")
    print("====================================================")
    
    # Parameters from the described experiment
    num_rounds = 20
    expected_dividend = 1.4  # Average of 1.2 and 1.6 with equal probability
    interest_rate = 0.05
    redemption_value = 28.0
    target_fundamental = 28.0
    
    print("\nExperiment Parameters:")
    print(f"Total Rounds: {num_rounds}")
    print(f"Expected Dividend: {expected_dividend}")
    print(f"Interest Rate: {interest_rate}")
    print(f"Redemption Value: {redemption_value}")
    print(f"Target Fundamental: {target_fundamental}")
    
    # Verify that the fundamental price is constant across periods
    print("\nFundamental Values by Period:")
    for period in range(1, num_rounds + 1):
        fv = calculate_fundamental_price(
            num_rounds, 
            expected_dividend, 
            interest_rate, 
            redemption_value,
            period
        )
        print(f"  Period {period}: {fv:.2f}")
    
    # Verify that the expected dividend yield equals the interest rate
    dividend_yield = expected_dividend / target_fundamental
    print(f"\nExpected Dividend Yield: {dividend_yield:.5f} ({dividend_yield*100:.2f}%)")
    print(f"Interest Rate: {interest_rate:.5f} ({interest_rate*100:.2f}%)")
    
    # Calculate the redemption value needed for a constant fundamental
    calculated_redemption = calibrate_redemption_value(
        num_rounds, 
        expected_dividend, 
        interest_rate, 
        target_fundamental
    )
    print(f"\nRequired Redemption Value: {calculated_redemption:.2f}")
    
    # Theoretical infinite horizon price (Gordon growth model without growth)
    infinite_price = expected_dividend / interest_rate
    print(f"\nInfinite Horizon Price (E(d)/r): {infinite_price:.2f}")

if __name__ == "__main__":
    main() 