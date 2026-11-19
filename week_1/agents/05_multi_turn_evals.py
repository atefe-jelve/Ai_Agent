# ==========================================
# Lesson 5: Multi-Turn Evaluations
# ==========================================


# 1. Mock database
PRODUCTS = {
    101: {
        "name": "Keyboard",
        "price": 200,
        "currency": "USD",
    },
    102: {
        "name": "Mouse",
        "price": 50,
        "currency": "USD",
    },
    103: {
    "name": "Monitor",
    "price": 300,
    "currency": "USD",
    "discount_percent": 10
}
}


# 2. Mock tools

def get_product_price(product_id):
    product = PRODUCTS.get(product_id)

    if product is None:
        return {"error": "Product not found"}

    return {
        "product_id": product_id,
        **product,
    }


def calculate_discount(price, discount_percent):
    final_price = price * (1 - discount_percent / 100)

    return {
        "original_price": price,
        "discount_percent": discount_percent,
        "final_price": final_price,
    }


# 3. Execute a predefined tool trajectory
def execute_trajectory(tool_calls):
    trace = []

    available_tools = {
        "get_product_price": get_product_price,
        "calculate_discount": calculate_discount,
    }

    for call in tool_calls:
        name = call["name"]
        arguments = call["arguments"]

        tool = available_tools.get(name)

        if tool is None:
            result = {"error": "Unknown tool"}
        else:
            try:
                result = tool(**arguments)
            except Exception as error:
                result = {"error": str(error)}

        trace.append({
            "tool": name,
            "arguments": arguments,
            "result": result,
        })

    return trace


# 4. Evaluation dataset

test_cases = [
    {
        "name": "Correct tool sequence",
        "tool_calls": [
            {
                "name": "get_product_price",
                "arguments": {"product_id": 101},
            },
            {
                "name": "calculate_discount",
                "arguments": {
                    "price": 200,
                    "discount_percent": 15,
                },
            },
        ],
        "expected_tools": [
            "get_product_price",
            "calculate_discount",
        ],
        "expected_final_price": 170,
    },
    {
        "name": "Wrong tool order",
        "tool_calls": [
            {
                "name": "calculate_discount",
                "arguments": {
                    "price": 200,
                    "discount_percent": 15,
                },
            },
            {
                "name": "get_product_price",
                "arguments": {"product_id": 101},
            },
        ],
        "expected_tools": [
            "get_product_price",
            "calculate_discount",
        ],
        "expected_final_price": 170,
    },
    {
        "name": "Incorrect discount",
        "tool_calls": [
            {
                "name": "get_product_price",
                "arguments": {"product_id": 101},
            },
            {
                "name": "calculate_discount",
                "arguments": {
                    "price": 200,
                    "discount_percent": 20,
                },
            },
        ],
        "expected_tools": [
            "get_product_price",
            "calculate_discount",
        ],
        "expected_final_price": 170,
    },
]


# 5. Evaluate a trajectory

def evaluate_trajectory(test_case):
    trace = execute_trajectory(test_case["tool_calls"])

    actual_tools = [
        step["tool"]
        for step in trace
    ]

    expected_tools = test_case["expected_tools"]

    # Check exact tool order
    order_passed = actual_tools == expected_tools

    # Check arguments
    expected_price = test_case["expected_final_price"]

    final_result = next(
        (
            step["result"]
            for step in trace
            if step["tool"] == "calculate_discount"
        ),
        None,
    )

    if final_result and "final_price" in final_result:
        actual_price = final_result["final_price"]

        price_passed = abs(
            actual_price - expected_price
        ) < 0.001
    else:
        actual_price = None
        price_passed = False

    passed = order_passed and price_passed

    return {
        "name": test_case["name"],
        "order_passed": order_passed,
        "price_passed": price_passed,
        "passed": passed,
        "actual_tools": actual_tools,
        "actual_price": actual_price,
        "trace": trace,
    }


# 6. Run all evaluations

def main():
    results = []

    for test_case in test_cases:
        result = evaluate_trajectory(test_case)

        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"

        print(f"\n[{status}] {result['name']}")

        print("Tool order:", result["actual_tools"])
        print("Order correct:", result["order_passed"])
        print("Final price:", result["actual_price"])
        print("Price correct:", result["price_passed"])

    passed_count = sum(
        result["passed"]
        for result in results
    )

    accuracy = passed_count / len(results) * 100

    print("\n============================")
    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()