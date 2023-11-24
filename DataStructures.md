# Data-Structures
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/40e9814b-54e9-417c-ad3b-90ecbd1ec388)

## *链表

```c++
struct ListNode {//单链表结点
	int val;
	ListNode* next;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(ListNode* next) :val(0), next(next) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}
};
```

```c++
struct DLNode {//双链表结点
	int val;
	DLNode* prior, * next;
	DLNode() :val(0), prior(nullptr), next(nullptr) {}
	DLNode(int x) : val(x), prior(nullptr), next(nullptr) {}
	DLNode(int x, DLNode* next) : val(x), prior(nullptr), next(next) {}
	DLNode(int x, DLNode* prior, DLNode* next) : val(x), prior(prior), next(next) {}
};
```

### 增

```c++
ListNode* head, * tail = &head;
//头插法
void insertAtHead(ListNode*& head, int val) {
    ListNode* newNode = new ListNode(val);
    newNode->next = head;
    head = newNode;
}
//尾插法
void insertAtTail(ListNode*& tail, int val) {
    ListNode* newNode = new ListNode(val);
    tail->next = newNode;
    tail = newNode;
}
```

在指定结点p之前插入

```c++
//在指定结点p之前插入
bool List::ListInsertPriorNode(ListNode* p, int e) {
	if (p == nullptr)//不合法
		return false;
	ListNode* s = new ListNode();
	s->next = p->next;
	p->next = s;
	s->val = p->val; //将p中元素复制到s中
	p->val = e;       // p中元素覆盖为e
	return true;
}
```

在第i个位置插入元素val

```c++
//在第i个位置插入元素val
bool List::ListInsert(ListNode*& L, int i, int val) {
	if (i < 1)
		return false;
	ListNode* p = GetElem(L, i - 1);//找到第i-1个结点 见查
	ListInsertPriorNode(p, val);
}
```

### 删

删除第i个位置元素

```c++
//删除第i个位置元素
bool List::ListDelete(ListNode*& L, int i) {
	if (i < 1)
		return false;
	ListNode* p = GetElem(L, i - 1);//找到第i-1个结点
	if (p == nullptr)//不合法
		return false;
	if (p->next == nullptr)
		return false;
	ListNode* q = p->next;//用q指向要删除的结点
	p->next = q->next;
	delete q;
	return true;
}
//删除倒数第n个位置结点 见算法前后指针
//删除链表中间的结点 见算法快慢指针
```



### 改

反转链表

```c++
//反转链表   206. 反转链表
ListNode* List::reverseList(ListNode* head) {
	//头结点存储数据
	ListNode* pre, * nex;
	pre = nullptr;
	while (head != nullptr) {
		nex = head->next;
		head->next = pre;
		pre = head;
		head = nex;
	}
	return pre;
}
```

反转链表区间

```c++
//反转链表区间  92. 反转链表 II
ListNode* List::reverseBetween(ListNode* head, int left, int right) {
	//头结点存储数据
	ListNode* dummy = new ListNode(head);//哨兵 下标为0
	ListNode* cur, * star, * lef, * rig, * end, * pre, * nex;
	cur = star = lef = rig = end = dummy;
	pre = nullptr;
	int number = 0;
	while (cur != nullptr) {
		if (number == left - 1) {
			star = cur;
			lef = cur->next;
		}
		if (number == right) {
			rig = cur;
			end = cur->next;
		}
		number++;
		cur = cur->next;
	}
	ListNode* temp = lef;
	while (temp != end) {//反转区间
		nex = temp->next;
		temp->next = pre;
		pre = temp;
		temp = nex;
	}
	star->next = rig;
	lef->next = end;
	if (left == 1)return star->next;
	return head;
}
```

归并两个有序链表

```c++
//二路归并两个链表
ListNode* MergeTwoLists(ListNode* a, ListNode* b) {
    if ((!a) || (!b)) return a ? a : b; //a和b中有一个为空，返回另一个
    ListNode head, * tail = &head, * aPtr = a, * bPtr = b;
    while (aPtr && bPtr) {//a,b都不空
        if (aPtr->val < bPtr->val) {
            tail->next = aPtr; aPtr = aPtr->next;
        }
        else {
            tail->next = bPtr; bPtr = bPtr->next;
        }
        tail = tail->next;
    }
    tail->next = (aPtr ? aPtr : bPtr);
    return head.next;
}
```



### 查

按位查找 查找链表第i个元素的值

```c++
//按位查找 查找链表第i个元素的值
ListNode* List::GetElem(ListNode* L, int i) {
	if (i < 0)
		return nullptr;
	ListNode* p;  //ָ指针p指向当前扫到的结点
	int count = 0;//记录p指向的是第几个结点，带头结点所以以0开始
	p = L;
	while (p != nullptr && count < i) {//循环找到第i个结点
		p = p->next;
		count++;
	}
	return p;
}
```

按值查找 查找链表中值为e的元素

```c++
//按值查找 查找链表中值为e的元素
ListNode* List::LocateElem(ListNode* L, int e) {
	ListNode* p = L->next;
	while (p != nullptr && p->val != e) {
		p = p->next;
	}
	return p;
}
```

## *栈

n个不同元素进栈，出栈元素不同排列的个数为一个卡特兰数：
```
(2n)!/(n +1)!n!
```
括号匹配问题 
### **单调栈
#### [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)
```c++
vector<int> dailyTemperatures(vector<int>& temperatures) {
	//从后向前
	vector<int>& te = temperatures;
	int n = te.size();
	vector<int> ans(n, 0);
	stack<int>s;
	for (int i = n - 1; i >= 0; i--) {
		while (!s.empty() && te[i] >= te[s.top()]) {
			s.pop();
		}
		if (!s.empty()) {//te[i]<s.top()
			ans[i] = s.top() - i;
		}
		//栈空 或者当日温度大于栈顶
		s.emplace(i);
	}
	return ans;
}
//从前向后
vector<int>& te = temperatures;
int n = te.size();
vector<int> ans(n, 0);
stack<int>s;
for (int i = 0; i < n; i++) {
	while (!s.empty() && te[i] > te[s.top()]) {
		ans[s.top()] = i - s.top();
		s.pop();
	}
	s.emplace(i);
}
/*while (!s.empty()) {
	ans[s.top()] = 0;
	s.pop();
}*/  //由于初始化ans为0 所以不需要这段
return ans;
```
#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)
```c++
int trap(vector<int>& height) {
	int n = height.size();
	stack<int>s;
	int ans = 0;
	for (int i = 0; i < n; i++) {
		while (!s.empty() && height[i] >= height[s.top()]) {
			int temp = s.top();
			s.pop();
			if (s.empty())break;
			ans += (i - s.top() - 1) * (min(height[s.top()], height[i]) - height[temp]);
		}
		s.emplace(i);
	}
	return ans;
}
```
## *队

```c++
struct SequenceQueue {
	int maxsize;
	int* data;
	int front, rear;
	SequenceQueue(int maxsize) {
		this->maxsize = maxsize;
		this->front = this->rear = 0;
		this->data = new int[maxsize];
	}
	~SequenceQueue() {
		delete[] data;
	}
};
```

```c++
struct LinkQueue {
	int val;
	LinkQueue* front, * rear, * next;
	LinkQueue() :val(0), next(nullptr), front(nullptr), rear(nullptr) {}
	LinkQueue(int x) :val(x), next(nullptr), front(nullptr), rear(nullptr) {}
};
```

### 入队

```c++
//入队
bool Queue::EnterQueue(SequenceQueue& Q, int x) {
	if ((Q.rear + 1) % Q.maxsize == Q.front)return false;//队满
	Q.data[Q.rear] = x;
	Q.rear = (Q.rear + 1) % Q.maxsize;
	return true;
}
bool Queue::EnterQueue(LinkQueue& Q, int x) {
	//不带头结点
	LinkQueue* elem = new LinkQueue(x);
	if (Q.front == nullptr) {//在空队插入第一个元素，修改队头队尾指针
		Q.front = elem;
		Q.rear = elem;
	}
	else {
		Q.rear->next = elem;
		Q.rear = elem;
	}
	return true;

	//带头结点
	/*LinkQueue* Elem = new LinkQueue(x);
	Q.rear->next = Elem;
	Q.rear = Elem;
	return true;*/
}
```

### 出队

```c++
//出队
int Queue::PopQueue(SequenceQueue& Q) {
	if (Q.rear == Q.front)return false;//队空
	int x = Q.data[Q.front];
	Q.front = (Q.front + 1) % Q.maxsize;
	return x;
}
bool Queue::PopQueue(LinkQueue& Q) {
	//不带头结点
	if (Q.front == nullptr) return false;//空队
	LinkQueue* x = Q.front;
	Q.front = x->next;
	if (Q.rear == x) Q.rear = Q.front = nullptr;//最后一个结点出队
	delete x;
	return true;

	//带头结点
	//if (Q.front == Q.rear) return false;//空队
	//LinkQueue* x = Q.front->next;
	//Q.front->next = x->next;
	//if (Q.rear == x) Q.rear = Q.front;//最后一个结点出队
	//delete x;
	//return true;
}
```
### **单调队
#### [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)
```c++
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	vector<int>ans;
	deque<int>q;//单调队列
	for (int i = 0; i < nums.size(); i++) {
		//入队
		while (!q.empty() && nums[q.back()] <= nums[i]) {
			q.pop_back();
		}
		q.emplace_back(i);
		//出队
		if (i - q[0] + 1 > k) {
			q.pop_front();
		}
		//记录答案
		if (i >= k - 1) {
			ans.emplace_back(nums[q.front()]);
		}
	}
	return ans;
}
```

## *哈希表

### [1. 两数之和](https://leetcode.cn/problems/two-sum/)

![1699109708486](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/16ba909e-8d2c-4d40-a322-465c26f7f52c)

```c++
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int>m;//创建一个空哈希表
    vector<int> ans;
    for (int i = 0; i < nums.size();i++) {// 枚举 i
        auto it = m.find(target - nums[i]);// 在左边找target-nums[i]
        if (it != m.end()) {// 找到了
            ans = { i,it->second };
            break;
        }
        m[nums[i]] = i;
    }
    return ans;
}
```

## *树

### **二叉树
```c++
//二叉树结点
struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};
```

```c++
//线索二叉树结点
struct ThreadTreeNode {
	int val;
	ThreadTreeNode* left;
	ThreadTreeNode* right;
	bool ltag, rtag;//true表示线索，false表示孩子
	ThreadTreeNode() :val(0), left(nullptr), right(nullptr), ltag(false), rtag(false) {}
	ThreadTreeNode(int x) : val(x), left(nullptr), right(nullptr), ltag(false), rtag(false) {}
};
```

#### 遍历二叉树

前序遍历

```c++
//前序遍历
vector<int> BinaryTree::preOrderTraversal(TreeNode* root) {
	vector<int>ans;
	std::function<void(TreeNode* root)> preOrder = [&](TreeNode* root) {
		if (root != nullptr) {
			ans.push_back(root->val);//根
			preOrder(root->left);//左
			preOrder(root->right);//右
		}
	};
	preOrder(root);
	return ans;
}
```

中序遍历

```c++
//中序遍历
vector<int> BinaryTree::inOrderTraversal(TreeNode* root) {
	vector<int>ans;
	std::function<void(TreeNode* root)> inOrder = [&](TreeNode* root) {
		if (root != nullptr) {
			inOrder(root->left);//左
			ans.push_back(root->val);//根
			inOrder(root->right);//右
		}
	};
	inOrder(root);
	return ans;
}
```

后序遍历

```c++
//后序遍历
vector<int> BinaryTree::postOrderTraversal(TreeNode* root) {
	vector<int>ans;
	std::function<void(TreeNode* root)> postOrder = [&](TreeNode* root) {
		if (root != nullptr) {
			postOrder(root->left);//左
			postOrder(root->right);//右
			ans.push_back(root->val);//根
		}
	};
	postOrder(root);
	return ans;
}
```

层序遍历

```c++
//层序遍历
vector<vector<int>> BinaryTree::levelOrderTraversal(TreeNode* root) {
	vector<vector<int>>ans;
	if (root == nullptr)return ans;
	queue<TreeNode* > q;
	q.push(root);
	std::function<void()> levelOrder = [&]() {
		vector<int>a;
		int N = q.size();
		while (N) {
			a.push_back(q.front()->val);
			if (q.front()->left != nullptr) {
				q.push(q.front()->left);//左
			}
			if (q.front()->right != nullptr) {
				q.push(q.front()->right);//右
			}
			q.pop();
			N--;
		}
		ans.push_back(a);
	};
	while (!q.empty()) {
		levelOrder();
	}
	return ans;
}
```

中序遍历加后序遍历确定唯一的二叉树（中序+其他任一遍历可唯一确定一颗二叉树）

```c++
//中序遍历加后序遍历确定唯一的二叉树
TreeNode* BinaryTree::buildTree(vector<int>& inorder, vector<int>& postorder) {
	int n = inorder.size();
	if (n == 0)return nullptr;
	map<int, int> m;
	for (int i = 0; i < n; i++) {
		m[inorder[i]] = i;
	}
	std::function<TreeNode* (int, int, int)> f =
		[&](int root, int left, int right)->TreeNode* {
		if (left > right)return nullptr;
		TreeNode* r = new TreeNode(postorder[root]);
		int R = m[postorder[root]];
		r->left = f(root - (right - R) - 1, left, R - 1);
		r->right = f(root - 1, R + 1, right);
		return r;
	};
	return f(n - 1, 0, n - 1);
}
```

遍历中序线索二叉树得到中序遍历

```c++
//遍历中序线索二叉树得到中序遍历
void BinaryTree::inOrderThreadTraversal(ThreadTreeNode* root) {
	std::function<ThreadTreeNode* (ThreadTreeNode*)> firstNode =
		[](ThreadTreeNode* p)->ThreadTreeNode* {
		//循环找到最左下结点
		while (!p->ltag) p = p->left;//找前驱类似
		return p;
	};
	std::function<ThreadTreeNode* (ThreadTreeNode*)> nextNode =
		[&](ThreadTreeNode* p)->ThreadTreeNode* {
		//右子树中最左下结点
		if (!p->rtag) return firstNode(p->right);//找前驱类似
		else return p->right;
	};
	for (ThreadTreeNode* p = firstNode(root); p != nullptr; p = nextNode(p))
		cout << p->val << " ";
	cout << endl;
}
```

#### 求树的高度

```c++
int BinaryTree::calculateH(TreeNode* root) {
	if (root == nullptr)return 0;
	int left_treeH = calculateH(root->left);
	int right_treeH = calculateH(root->right);
	return max(left_treeH, right_treeH) + 1;
}
```
#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
```c++
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
	int n = inorder.size();
	if (n == 0)return nullptr;
	map<int, int> m;
	for (int i = 0; i < n; i++) {
	    m[inorder[i]] = i;
	}
	function<TreeNode* (int, int,int)> f = [&](int root,int left, int right)->TreeNode* {
	    if (left > right)return nullptr;
	    TreeNode* r = new TreeNode(postorder[root]);
	    int R = m[postorder[root]];
	    r->left = f(root - (right - R) - 1, left, R - 1);
	    r->right = f(root - 1, R + 1, right);
	    return r;
	};
	return f(n - 1, 0, n - 1);
}
```
#### [1008. 前序遍历构造二叉搜索树](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)
```c++
TreeNode* bstFromPreorder(vector<int>& preorder) {
	int n = preorder.size();
	function<TreeNode* (int, int)>dfs = [&](int left, int right)->TreeNode* {
		if (left > right)return nullptr;
		TreeNode* root = new TreeNode(preorder[left]);
		int i;
		for (i = left + 1; i <= right; i++) {
			if (preorder[i] > preorder[left])break;
		}
		root->left = dfs(left + 1, i - 1);
		root->right = dfs(i, right);
		return root;
	};
	return dfs(0, n - 1);
}
```
#### [538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/)
```c++
TreeNode* convertBST(TreeNode* root) {
	int sum = 0;
	function<void(TreeNode*)>dfs = [&](TreeNode* root) {
		if (root == nullptr)return;
		dfs(root->right);
		sum += root->val;
		root->val = sum;
		dfs(root->left);
	};
	dfs(root);
	return root;
}
```
#### [450. 删除二叉搜索树中的节点](https://leetcode.cn/problems/delete-node-in-a-bst/)
```c++
TreeNode* deleteNode(TreeNode* root, int key) {
	if (root == nullptr)return nullptr;
	if (root->val > key) {
		root->left = deleteNode(root->left, key);
	}
	else if (root->val < key) {
		root->right = deleteNode(root->right, key);
	}
	else {//val==key
		if (!root->left && !root->right)return nullptr;
		else if (!root->left)return root->right;
		else if (!root->right)return root->left;
		else {
			TreeNode* new_root = root->right;
			while (new_root->left) {
				new_root = new_root->left;
			}
			root->right = deleteNode(root->right, new_root->val);
			new_root->right = root->right;
			new_root->left = root->left;
			return new_root;
		}
	}
	return root;
}
```

### **堆

堆是一个完全二叉树

大根堆：每个节点的值都大于或者等于他的左右孩子节点的值

小根堆：每个节点的值都小于或者等于他的左右孩子节点的值

### **树状数组
#### [307. 区域和检索 - 数组可修改](https://leetcode.cn/problems/range-sum-query-mutable/)
```c++
class NumArray {
private:
	vector<int> nums;
	vector<int> tree;

	int prefixSum(int i) {
		int t = 0;
		for (; i > 0; i &= i - 1) { // i -= i & -i 的另一种写法
			t += tree[i];
		}
		return t;
	}
public:
	NumArray(vector<int>& nums) {
		this->nums.resize(nums.size());
		this->tree.resize(nums.size() + 1);
		for (int i = 0; i < nums.size(); i++) {
			update(i, nums[i]);
		}
	}

	void update(int index, int val) {
		int delta = val - nums[index];
		nums[index] = val;
		for (int i = index + 1; i < tree.size(); i += i & -i) {
			tree[i] += delta;
		}
	}

	int sumRange(int left, int right) {
		return prefixSum(right + 1) - prefixSum(left);
	}
};
```

### **字典树/前缀树
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/05b88cea-d07f-46c2-a64b-2b25016a1e24)

#### [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)
```c++
class Trie {
private:
	vector<Trie*> children;//第i位不是nullptr表示第i个字母存在
	bool isEnd;//表示这一位是不是单词的结尾

	Trie* searchPrefix(string prefix) {//找到prefix在字典树的前缀
		Trie* node = this;
		for (char ch : prefix) {
			ch -= 'a';
			if (node->children[ch] == nullptr) {
				return nullptr;
			}
			node = node->children[ch];
		}
		return node;
	}

public:
	Trie() : children(26), isEnd(false) {}

	void insert(string word) {
		Trie* node = this;
		for (char ch : word) {
			ch -= 'a';
			if (node->children[ch] == nullptr) {
				node->children[ch] = new Trie();
			}
			node = node->children[ch];
		}
		node->isEnd = true;
	}

	bool search(string word) {
		Trie* node = this->searchPrefix(word);
		return node != nullptr && node->isEnd;
	}

	bool startsWith(string prefix) {
		return this->searchPrefix(prefix) != nullptr;
	}
};
```
#### [2416. 字符串的前缀分数和](https://leetcode.cn/problems/sum-of-prefix-scores-of-strings/)
```c++
```
