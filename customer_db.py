



name_dict = {"Customer1": "danial"}


def check(customer):
    # Return the name if customer is present in the dictionary, else return False
    return name_dict.get(customer, False)


if __name__ == "__main__":
    print(check("Customer1"))  # Example usage