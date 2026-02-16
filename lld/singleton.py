class Logger():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

log1 = Logger()
log2 = Logger()
print(log1==log2)


# strategy pattern
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using Credit Card")

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using PayPal")

# context class
class PaymentContext:
    def __init__(self, strategy:PaymentStrategy):
        self._strategy = strategy
    def pay(self, amount):
        self._strategy.pay(amount)

# usage
Context = PaymentContext(CreditCardPayment())
Context.pay(100)

Context = PaymentContext(PayPalPayment())
Context.pay(200)

class PaymentFactory:
    @classmethod
    def create_payment_method(cls, payment_type: str):
        if payment_type == "credit_card":
            return CreditCardPayment()
        else:
            return PayPalPayment()
# usage
payment_method = PaymentFactory().create_payment_method("credit_card")
payment_method.pay(100)

payment_method = PaymentFactory().create_payment_method("paypal").pay(200)


class SingleTon:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance