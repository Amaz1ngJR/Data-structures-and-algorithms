# 位运算
```c++
bool demo() {
	int num = 100;
	for (int i = 0; i < 31; i++) {
		int a = (num >> i) & 1;//a表示num的第i位是否是1
		int mask = 1 << i;//设置一个第i位的掩码
		int result = (num & mask) >> i;//单独将num的第i位取出来

		int ans = 0;
		ans |= 1 << i;//将ans的第i位置为1
		ans &= ~(1 << i);//将ans的第i位置为0
		ans ^=  (1 << i);//将ans的第i位置为0
	}
	while (n) {
		cout << n << endl;
		n &= n - 1; //去掉n二进制中低位第一个1 n -= n & -n 的另一种写法
	}
	return (n&(n-1))==0;//判断一个正整数是否是2的幂
}
```

[2917. 找出数组中的 K-or 值](https://leetcode.cn/problems/find-the-k-or-of-an-array/)

```c++
int findKOr(vector<int>& nums, int k) {
    int ans = 0;
    for (int i = 0; i < 31; i++) {//元素范围最多31位
        int cnt1 = 0;//第i位是1的元素的个数
        for (const auto& v : nums) {
            cnt1 += (v >> i) & 1;//元素右移i位 判断第i位是否是1
        }
        if (cnt1 >= k) {
            ans |= 1 << i;//将1左移i位与ans| 将第i位的1添加到ans上
        }
    }
    return ans;
}
```
获得数字的每一位
```c++
for (const auto& v : nums) {
	string temp(31, '0');//用长度为31的字符串记录元素的每一位
	for (int i = 0; i < 31; i++) {
		if (v >> i & 1)temp[30 - i] = '1';
	}
	std::bitset<64> bits(temp);//#include <bitset> 将二进制字符串表示为二进制
	int num = bits.to_ulong();//将二进制转成十进制
}
void test() {
	int a = 7;
	//使用std::bitset #include<bitset>将整数转换为二进制字符串
	std::bitset<32> bita(a);
	for (int i = 0; i < bita.size(); i++) {
		cout << bita[i];
	}
	cout<<endl;
	cout<<bita<<endl;
}
```
[190. 颠倒二进制位](https://leetcode.cn/problems/reverse-bits/)
```c++
uint32_t reverseBits(uint32_t n) {
	uint32_t ans = 0;
	for (int i = 0; i < 32 && n; i++) {
		ans |= (n & 1) << (31 - i);
		n >>= 1;
	}
	return ans;
}
```
[201. 数字范围按位与](https://leetcode.cn/problems/bitwise-and-of-numbers-range/)
```c++
int rangeBitwiseAnd(int left, int right) {
	/*bitset<32>num(left), mask(0);
	while (num != mask && left < right) {
		num &= bitset<32>(++left);
	}
	return num == mask ? 0 : num.to_ulong();*/ //超时代码
	int shift = 0;
	while (left < right) {
		left >>= 1;
		right >>= 1;
		shift++;
	}
	return left << shift;
}
```
判断一个数是否是奇数
```c++
if(n % 2 == 1) => if(n & 1)
```
判断两个数是否同号
```c++
if((a ^ b) < 0)
```
判断i--是否小于0了
```c++
for(int i = n;i >= 0; i--) => for(int i = n; ~i; i--) //~(-1) == 0
```
交换两个整数的值
```c++
int a = 10, b = 100;
a ^= b;
b ^= a;
a ^= b; std::cout << "a = " << a << " b = " << b << std::endl;
```
## bitset < bitset >
```c++
void demo() {
	//初始化
	std::bitset<8> a(42);//使用整数值 42 初始化一个包含 8 位的 bitset
	std::bitset<4> myBits("1010");  // 使用二进制字符串 "1010" 初始化一个包含 4 位的 bitset
	bool bitValue = myBits[2];  // 获取第 2 位的值
	myBits <<= 2;  // 将所有位左移两位
	//成员函数
	int size = myBits.size();//获取 bitset 的大小/位数
	int n = myBits.count();//获取二进制中1的个数
	bool isSet = myBits.test(3);//检查第 3 位是否被设置为1
	myBits.set(2);    // 将第 2 位设置为 1 myBits |= (1 << 2);
	myBits.reset(4);  // 将第 4 位重置为 0 myBits &= ~(1 << 4);
	myBits.flip(1);  // 将第 1 位的值取反
	std::string bitString = myBits.to_string();  // 将 bitset 转换为二进制字符串
	int num = myBits.to_ulong();//将 bitset 二进制转成十进制
}
```
[37. 解数独](https://leetcode.cn/problems/sudoku-solver/)
```c++
void solveSudoku(vector<vector<char>>& board) {
	vector<bitset<9>>rows(9, bitset<9>(0));
	vector<bitset<9>>cols(9, bitset<9>(0));
	vector<vector<bitset<9>>>cells(3, vector<bitset<9>>(3, bitset<9>(0)));
	function<bitset<9>(int, int)>status = [&](int x, int y)->bitset<9> {//当前位置的所有可能选择的数
		return ~(rows[x] | cols[y] | cells[x / 3][y / 3]);
	};
	function<vector<int>()>next = [&]()->vector<int> {//下一个最少回溯的位置
		vector<int>ret(2); int min_ = 10, cunt;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] == '.') {
					cunt = status(i, j).count();
					if (cunt < min_) {
						ret = { i,j };
						min_ = cunt;
					}
				}
			}
		}
		return ret;
	};
	function<void(int, int, int, bool)>fill = [&](int x, int y, int n, bool flag) {
		rows[x][n] = flag;
		cols[y][n] = flag;
		cells[x / 3][y / 3][n] = flag;
	};
	function<bool(int)>dfs = [&](int cnt)->bool {
		if (!cnt)return true;
		auto it = next();
		auto stat = status(it[0], it[1]);
		for (int i = 0; i < 9; i++) {//(it[0],it[1])的位置
			if (stat.test(i)) {//当前位置的值为i + '1'
				fill(it[0], it[1], i, true);
				board[it[0]][it[1]] = i + '1';
				if (dfs(cnt - 1))return true;//剪枝
				board[it[0]][it[1]] = '.';
				fill(it[0], it[1], i, false);
			}
		}
		return false;
	};
	int cnt = 0, n;
	for (int i = 0; i < 9; i++) {//init
		for (int j = 0; j < 9; j++) {
			cnt += (board[i][j] == '.');
			if (board[i][j] != '.') {
				n = board[i][j] - '1';
				rows[i].set(n); // rows[i] |= (1 << n);
				cols[j].set(n); // cols[j] |= (1 << n);
				cells[i / 3][j / 3].set(n);// cells[i / 3][j / 3] |= (1 << n);
			}
		}
	}
	dfs(cnt);
}
```
```c++
void solveSudoku(vector<vector<char>>& board) {
	vector<bitset<9>>rows(9, bitset<9>(0)), cols(9, bitset<9>(0));
	vector<vector<bitset<9>>>cells(3, vector<bitset<9>>(3, bitset<9>(0)));
	function<vector<int>()>next = [&]()->vector<int> {//下一个最少回溯的位置
		vector<int>ret(2); int min_ = 10, cunt;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] == '.') {
					cunt = (~(rows[i] | cols[j] | cells[i / 3][j / 3])).count();
					if (cunt < min_) {
						ret = { i,j };
						min_ = cunt;
					}
				}
			}
		}
		return ret;
	};
	function<bool(int)>dfs = [&](int cnt)->bool {
		if (!cnt)return true;
		auto it = next();
		auto status = ~(rows[it[0]] | cols[it[1]] | cells[it[0] / 3][it[1] / 3]);
		for (int i = 0; i < 9; i++) {//(it[0],it[1])的位置
			if (status.test(i)) {//当前位置的值为i + '1'
				rows[it[0]][i] = cols[it[1]][i] = cells[it[0] / 3][it[1] / 3][i] = true;
				board[it[0]][it[1]] = i + '1';
				if (dfs(cnt - 1))return true;//剪枝
				board[it[0]][it[1]] = '.';
				rows[it[0]][i] = cols[it[1]][i] = cells[it[0] / 3][it[1] / 3][i] = false;
			}
		}
		return false;
	};
	int cnt = 0, n;
	for (int i = 0; i < 9; i++) {//init
		for (int j = 0; j < 9; j++) {
			cnt += (board[i][j] == '.');
			if (board[i][j] != '.') {
				n = board[i][j] - '1';
				rows[i] |= (1 << n);
				cols[j] |= (1 << n);
				cells[i / 3][j / 3] |= (1 << n);
			}
		}
	}
	dfs(cnt);
}
```
## __builtin GCC编译器
```c++
void demo() {
	int count = __builtin_popcount(255);//返回整数中二进制表示中1的个数 8
	int trailing_zeros = __builtin_ctz(16);//返回整数的二进制表示中从最低位开始到第一个非零位的位数4
	int leading_zeros = __builtin_clz(16);//返回整数的二进制表示中从最高位开始到第一个非零位的位数 28
}
```
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/089180b1-3747-42f2-97fb-db4951eea74a)

[137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)
```c++
int singleNumber(vector<int>& nums) {
	int ans = 0;
	for (int i = 0; i < 32; ++i) {
		int cnt = 0;
		for (const int& v : nums)
			cnt += v >> i & 1;
		ans |= cnt % 3 << i;
	}
	return ans;
}
```
# 模运算
[2575. 找出字符串的可整除数组](https://leetcode.cn/problems/find-the-divisibility-array-of-a-string/)

c++
```c++
vector<int> divisibilityArray(string word, int m) {
	int n = word.size();
	vector<int>ans(n);
	long long x = 0;
	for (int i = 0; i < n; ++i) {
		x = (10 * x + (word[i] - '0')) % m;
		ans[i] = x == 0;
	}
	return ans;
}
```
python
```python
def divisibilityArray(self, word: str, m: int) -> List[int]:
	ans = []
        x = 0
        for i in map(int, word):
            x = (10 * x + i) % m
            ans.append(0 if x else 1)
        return ans
```
# 重叠区间、区间合并
[452. 用最少数量的箭引爆气球](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)
```c++
int findMinArrowShots(vector<vector<int>>& points) {
	//按起点靠前排序 起点相同按终点靠前排序
	sort(points.begin(), points.end(),
	    [&](const auto& a, const auto& b) {
		if (a[0] == b[0]) return a[1] < b[1];
		return a[0] < b[0]; });
	int n = points.size(), low = points[0][0], high = points[0][1], ans = 1;
	for (int i = 1; i < n; i++) {//从左向右遍历寻找重叠区间
	    if (points[i][0] <= high) {//区间和左侧区间有重叠 
		low = points[i][0];//更新重叠区间
		high = min(high, points[i][1]); //更新重叠区间
	    }
	    else {//没有重叠
		ans++;
		low = points[i][0];
		high = points[i][1];
	    }
	}
	return ans;
}
```
do [2580. 统计将重叠区间合并成组的方案数](https://leetcode.cn/problems/count-ways-to-group-overlapping-ranges/)
[56. 合并区间](https://leetcode.cn/problems/merge-intervals/)
```c++
vector<vector<int>> merge(vector<vector<int>>& intervals) {
	vector<vector<int>> ans;
	if (intervals.empty()) return ans;
	sort(intervals.begin(), intervals.end(),
		[](const vector<int>& a, const vector<int>& b)->bool {
			if (a[0] == b[0])return a[1] < b[1];
			return a[0] < b[0];
		});
	ans.emplace_back(intervals[0]);
	for (int i = 1; i < intervals.size(); i++) {
		if (intervals[i][0] <= ans.back()[1]) {//当前区间的左端点小于数组末的区间的右端点
			ans.back()[1] = max(ans.back()[1], intervals[i][1]);//更新区间右端点
		}
		else
			ans.emplace_back(intervals[i]);
	}
	return ans;
}
```
[55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)
```c++
bool canJump(vector<int>& nums) {
	int maxright = nums[0];//maxright为目前所能跳的最远距离
	int i = 1, n = nums.size();
	while (i <= maxright && maxright < n) {//遍历所能跳的距离 更新所能跳的最大距离
		maxright = max(maxright, nums[i] + i);
		i++;
	}
	return maxright >= n - 1;
}
```
隔板的插入种数
[2963. 统计好分割方案的数目](https://leetcode.cn/problems/count-the-number-of-good-partitions/)
```c++
int numberOfGoodPartitions(vector<int>& nums) {
	//合并区间：1、遍历数组 维护每个元素最后一次出现的下标
	//2、再次遍历数组 合并区间
	unordered_map<int, int>m; int mod = 1000000007;
	for (int i = 0; i < nums.size(); i++) m[nums[i]] = i;
	int max_right = 0, ans = 1;
	for (int i = 0; i < nums.size() - 1; i++) {//少选最后一段区间使得n-1
		max_right = max(max_right, m[nums[i]]);
		if (i == max_right) {//达到区间的最大右端点 相当于合并了一个区间
			//n++;//记录合并后的区间个数
			//对n个数进行分割最多插入n-1个隔板分成n个
			//最少插入0个隔板分成1个
			//也就是有n-1个隔板 每个隔板有 选或不选 两个状态 即pow(2,n-1)
			ans = ans * 2 % mod;
		}
	}
	return ans;
}
```
# 数学
## 快速幂
[50. Pow(x, n)](https://leetcode.cn/problems/powx-n/)
```c++
double myPow(double x, int n) {
    if(n==1) return x;
    else if(n==-1) return 1/x;
    else if(n==0) return 1;
    double tmp = myPow(x,n/2);
    return tmp*tmp*myPow(x, n%2);
}
```
## 最大公因子/最大公约数
辗转相除法(欧几里得算法)得到两个数的最大公因数
```
用较大的数除以较小的数 余数为0则较小的数为最大公因数 余数不为0 则余数和较小的数重复上述
例：36和24的最大公因数
36 % 24 == 12 != 0 => 24 % 12 ==0 =>最大公约数为min(24,12)
```
```c++
function<int(int, int)>gcd = [&](int x, int y)->int {
	int res;
	while (y != 0) {
		res = y;
		y = x % y;
		x = res;
	}
	return x;
};
function<int(int, int)>gcd2 = [&](int x, int y)->int {
	if (x > y) return gcd2(y, x);
	if (y % x == 0) return x;
	return gcd2(y % x, x);
};
function<int(int, int)>gcd3 = [&](int x, int y)->int {
	return y ? gcd3(y, x % y) : x;
};
```
## 求质数
[204. 计数质数](https://leetcode.cn/problems/count-primes/)
```c++
int countPrimes(int n) {
	if (n == 0 || n == 1)return 0;
	vector<int>cnt(n);//0表示质数 1表示非质数 
	cnt[0] = cnt[1] = 1;//0和1是非质数
	for (int i = 2; i * i < n; i++) {
		if (!cnt[i]) {//i是质数
			for (int j = i * i; j < n; j += i) {//i的倍数都是非质数
				cnt[j] = 1;
			}
		}
	}
	return n - accumulate(begin(cnt), end(cnt), 0);
}
```
## 不等式
均值(算术-几何平均值)不等式
```
(x1 + x2 + ... + xn) / n >= (x1*x2*...xn)^(1 / n) 当且仅当x1 = x2 = ... xn取等
=> S >= n * (x1*x2*...xn)^(1 / n)
```
[343. 整数拆分](https://leetcode.cn/problems/integer-break/)
```c++
int integerBreak(int n) {
	int ans = 0;
	function<int(int, int)>func = [&](int num, int k)->int {
		int res = 1;
		while (k > 0) {//将num"平均"分成了k个数 并相乘
			res *= (num / k);
			num -= num / k;
			k--;
		}
		return res;
	};
	for (int k = 2; k <= n; k++) {
		ans = max(ans, func(n, k));
	}
	return ans;
}
```
柯西不等式
```
(a1^2 + a2^2 + ... + an^2) * (b1^2 + b2^2 + ... + bn^2) >= (a1*b1 + a2*b2 + ... + an*bn)
当且仅当 a1/b1 = a2/b2 = ... = an/bn 或 b1 = b2 = ... = bn = 0取等
```
阿姆斯特朗不等式
```
a^2 + b^2 + c^2 >= a*b +b*c +c*a   当且仅当a = b = c 取等
```
雅可比不等式
```
(a + b + c)^2 >= 3(a*b + b*c + c*a) 当且仅当a = b = c 取等
```
排序不等式
```
两个递增序列a1, ..., an; b1, ..., bn; ci是bi的乱序 满足：sum(ai*ci)最大是ai与bi正序乘 最小是ai与bi倒序乘 等号在ai==bi取等
```
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/8c9d035c-6650-40c9-a658-2de0ec425b36)
