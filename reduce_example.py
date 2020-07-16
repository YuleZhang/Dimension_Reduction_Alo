from functools import reduce
def add_5(value: int) -> int:
    return value + 5

def add_6(value: int) -> int:
    return value + 6

def add_7(value: int) -> int:
    return value + 7

y = reduce(
    lambda value, function: function(value),
    (
        add_5,
        add_6,
        add_7,
    ),
    1,
)

print(y)