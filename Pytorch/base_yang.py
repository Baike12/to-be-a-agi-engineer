import torch
def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        print("left",left)
        right = x[:n]
        print("right", right)
    elif n > 0:
        right = x[-1].repeat(n,1)
        left = x[n:]
    else:
        return x
    b = torch.cat((left, right), dim=0)
    print("b",b)
    return b


def concat_feat1(x, concat_n):
    assert concat_n % 2 == 1 # n 必须是奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    print("origin x:",x.shape)
    print("origin x:",x)
    x = x.repeat(1, concat_n)
    print("after repeat:",x.shape)
    print("after repeat:",x)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    print("after permute:",x.shape)
    print("after permute:",x)
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        print(mid,r_idx)
        print("before shift x[mid + r_idx, :]", x[mid + r_idx, :])
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        print("after shift x[mid + r_idx, :]", x[mid + r_idx, :])
        print("before shift x[mid - r_idx, :]", x[mid - r_idx, :])
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)
        print("after shift x[mid - r_idx, :]", x[mid - r_idx, :])
    print("after shift", x)
    x = x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)
    print("finish concat",x.shape)
    print("finish concat",x)
    return x


def test_concatn():
    a = torch.arange(1,91)
    a = a.reshape([3,6,5])
    # print(a[1])
    # print(a[1].shape)
    b = concat_feat1(a[1], 5)
    # print(b)
    # print(b)

def testtensor():
    a = torch.arange(1,91)
    a = a.reshape([3,6,5])
    b = a[0]
    b[1] = 0
    print(b)
    print(a)

def main():
    test_concatn()
    # testtensor()

if __name__ == "__main__":
    main()