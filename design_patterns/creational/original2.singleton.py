class Shape:
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        return "Drawing a Circle"

class Square(Shape):
    def draw(self):
        return "Drawing a Square"

class ShapeFactory:
    @staticmethod
    def get_shape(shape_type):
        if shape_type == "circle":
            return Circle()
        elif shape_type == "square":
            return Square()
        else:
            raise ValueError("Unknown shape type")

# Usage
factory = ShapeFactory()
shape1 = factory.get_shape("circle")
print(shape1.draw())  # Output: Drawing a Circle

shape2 = factory.get_shape("square")
print(shape2.draw())  # Output: Drawing a Square