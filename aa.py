
A = [4,9,8,2,6]
K = 3
def sol(A, K):
    A.sort()
    even = []
    odd = []
    temp_odd = -1
    for item in A:
        if(item % 2 == 0):
            even.append(item)
        elif(temp_odd != -1):
            odd.append(item + temp_odd)
            temp_odd = -1
        else:
            temp_odd = item
    k = K
    ret_sum = 0
    while( (k >= 2) and (len(even)>= 2) and (len(odd)>= 1)):
        even_item1 = even.pop()
        even_item2 = even.pop()
        odd_item = odd.pop()
        if( (even_item1 + even_item2 ) > odd_item):
            ret_sum += even_item1
            even.append(even_item2)
            odd.append(odd_item)
            k -= 1
        else:
            ret_sum += odd_item
            even.append(even_item2)
            even.append(even_item1)
            k -= 2
    if( k == 2) and (len(even) >= 2):
        ret_sum += even.pop()
        ret_sum += even.pop()
    elif( k == 2) and (len(odd) >= 1):
        ret_sum += odd.pop()
    elif (k == 1) and (len(even) >= 1):
        ret_sum += even.pop()
    elif( k == 1):
        return -1
    return ret_sum


if __name__ == '__main__':
    print(sol(A,K))

    print('aa')