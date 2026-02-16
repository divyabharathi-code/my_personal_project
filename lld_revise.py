class Shape:
    @abstractmethod
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        print("Drawing a Circle")

class Square(Shape):
    def draw(self):
        print("Drawing a Square")

class ShapeFactory:
    def get_shape(self, shape_type):
        if shape_type == "circle":
            return Circle()
        elif shape_type == "square":
            return Square()
        else:
            return None

# Usage
factory = ShapeFactory()
shape1 = factory.get_shape("circle")
shape1.draw()  # Output: Drawing a Circle

shape2 = factory.get_shape("square")
shape2.draw()  # Output: Drawing a Square



class PaymentStrategy:
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using Credit Card.")

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        print(f"Paid {amount} using PayPal.")

class PaymentContext:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def pay(self, amount):
        self.strategy.pay(amount)

# Usage
context = PaymentContext(CreditCardPayment())
context.pay(100)  # Paid 100 using Credit Card.

context.set_strategy(PayPalPayment())
context.pay(200)  # Paid 200 using PayPal.
