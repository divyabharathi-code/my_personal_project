from typing import List

from django.template.defaultfilters import length

products = [
    {'id': 1, 'name': 'Laptop', 'category': 'Electronics', 'price': 999.99, 'quantity': 5},
    {'id': 2, 'name': 'Smartphone', 'category': 'Electronics', 'price': 699.99, 'quantity': 10},
    {'id': 3, 'name': 'T-shirt', 'category': 'Clothing', 'price': 19.99, 'quantity': 100},
    {'id': 4, 'name': 'Jeans', 'category': 'Clothing', 'price': 49.99, 'quantity': 50},
    {'id': 5, 'name': 'Blender', 'category': 'Home Appliances', 'price': 39.99, 'quantity': 25}
]

def process_product(products):
    out_put_map = {}
    for i in range(len(products)):
        p = products[i]
        if 'category' in products[i]:
            temp_category = p.get('category')
            total_value = p['price'] * p['quantity']
            result = p
            result['total_value'] = total_value
            if temp_category in out_put_map:
                out_put_map[temp_category].append(result)
            else:
                out_put_map[temp_category] = [result]
    return out_put_map

# print(process_product(products))

# class Solution:
#     def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
#         l = 0
#         output = [[nums1[0], nums2[0]]]
#         i = 0
#         j = 0
#         while l <k and i < len(nums1)-1:
#             if nums1[i] < nums2[j] and j < len(nums2)-1:
#                 j += 1 if j < len(nums2) - 1 else 0
#             else:
#                 i += 1
#                 j = 0
#             l += 1
#             output.append([nums1[i], nums2[j]])
#         return output
# Solution().kSmallestPairs(nums1 = [1,2,4,5,6], nums2 = [3,5,7,9], k = 6)


class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        # it is goint to always odd leng
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] != nums[mid + 1] and nums[mid] != nums[mid - 1]:
                return nums[mid]
            elif nums[mid] == nums[mid] + 1:
                left = mid + 1
            elif nums[mid] == nums[mid - 1]:
                right = mid - 1

# Solution().singleNonDuplicate([1,1,2,3,3,4,4,8,8])


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = 0
        n = len(nums)-1
        for i in range(1, n):
            if nums[i] != nums[i-1]:
                l += 1
                nums[l] = nums[i-1]
            else:
                continue
        return l + 1, nums

# print(Solution().removeDuplicates([0,0,1,1,1,2,2,3,3,4]))

class Solution:
    def compress(self, chars: List[str]) -> int:
        l = 0
        result = []
        while l < len(chars):
            count = 0
            while l < len(chars)-1 and chars[l] == chars[l+1]:
                count +=1
                l +=1
            result.append(chars[l])
            if count > 0:
                result.append(count)
            l +=1
        return len(result)

class Solution:
    def compress(self, chars: List[str]) -> int:
        write = 0
        read = 0

        while read < len(chars):
            char = chars[read]
            count = 0

            while read < len(chars) and chars[read] == char:
                read += 1
                count += 1

            chars[write] = char
            write += 1

            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1

        return write

# print(Solution().compress(["a","a","b","b","c","c","c"]))

class Codec:
    def encode(self, strs:[str]) -> str:
        """encodes string"""
        encoded = ""
        for s in strs:
            encoded += f"{str(len(s))}#{s}"
        return encoded

    def decode(self, s: str) -> [str]:
        decoded = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != '#':
                j +=1
            length = s[i:j]
            word = s[j+1: j+1+length]
            decoded.append(word)
            i = j +1+length

s = "divya#bharathi#"
print(s.find("#", 6))