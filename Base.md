# 快速幂
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
判断一个数是否是奇数
```c++
if(n % 2 == 1) => if(n & 1)
```
# 区间合并
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
[100136. 统计好分割方案的数目](https://leetcode.cn/problems/count-the-number-of-good-partitions/)
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
