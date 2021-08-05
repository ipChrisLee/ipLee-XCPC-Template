# 常用定义

```c++
typedef long long int ll;
typedef double db;
const db EPS=1e-7; // 有时候也用eps
```



# 一般技巧

## 输入输出

### `iostream`的设置

* 浮点数精度设置

  ```c++
  cout<<fixed<<setprecision(15);
  ```

* 关闭流同步

  `ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);`



### 非负整数快读

```c++
template<typename I>I qread(){
    I x=0;char ch=getchar();
    while(ch<'0' || ch>'9'){ ch=getchar(); }
    while(ch>='0' && ch<='9'){ x=x*10+ch-'0';ch=getchar(); }
    return x;
}
```



### 带负整数快读

```c++
template<typename I>I qread(){
    I x=0,f=1;char ch=getchar();
	while (ch<'0'||ch>'9'){if (ch=='-') f=-1;ch=getchar();}
	while (ch>='0'&&ch<='9'){x=x*10+ch-'0';ch=getchar();}
	return x*f;
}
```



## 二分

### 整数二分

对$[1,n]$​区间整数二分，$work(x)$​是一个函数，$x$​较大的时候为`true`，较小时为`false`，输出最小的令$work(x)$为`true`的$x$，如果不存在输出`0`

```c++
int il=1,ir=n,imid,re=0;
while(il<=ir){
    imid=(il+ir)>>1;
    if(work(imid)){
        re=imid;ir=imid-1;
    }else{
        il=imid+1;
    }
}
```



### 实数二分

对$[l,r]$​​​区间实数二分，找$work(x)$​​​为`true`最小的点，小于答案的$work$​​​为`false`，大等于的为`true`

```c++
db mid;
while(r-l>EPS){
    mid=(l+r)/2;
    if(work(mid)){
        r=mid;
    }else{
        l=mid;
    }
}
// 之后使用l或者r都行。。。反正区间范围很小了
```



### `STL`写法

```c++
lower_bound(a+1,a+n+1,num); // 返回指向a[1...n]中第一个大于等于num的数字的指针
upper_bound(a+1,a+n+1,num); // 返回指向a[1...n]中第一个大于num的数字的指针
// 如果不存在返回a+n+1
// 可以通过upper_bound查找小于等于某个数的元素数量
```





## 三分

### 整数三分

在区间$[l,r]$​上找类二次函数$f(x)$​​的极小整点，注意整数三分取不到边界点，需要特判

```c++
double tri(){
    int l=1,r=1e9;
    while(r-l>5){
        int lmid=(1ll*l*2+r)/3;
        int rmid=(1ll*l+r*2)/3;
        double flmid=f(lmid),frmid=f(rmid);
        if(flmid<=frmid){
            r=rmid;
        }else{
            l=lmid;
        }
    }
    double ans=f(l);
    for(int i=l+1;i<=r;++i){
        ans=min(f(i),ans);
    }
    return ans
}
```



### 实数三分

实数三分更简单些，也不需要考虑端点问题

这里求的是$f(x)$的极大值点

```c++
db l=0,r=1e9,lmid,rmid; // 根据需要取
while(r-l>eps){
    lmid=(l*2+r)/3;
    rmid=(l+r*2)/3;
    if(f(lmid)>f(rmid)){
        r=rmid;
    }else{
        l=lmid;
    }
}
// 之后用l或r都行
```



## 离散化

```c++
const int MAXN=1e5+10;
int v[MAXN],vd[MAXN],n,tot;
void lsh(){
    for(int i=1;i<=n;++i){
        vd[i]=v[i];
    }
    sort(vd+1,vd+n+1);
    tot=unique(vd+1,vd+n+1)-vd-1;
    for(int i=1;i<=n;++i){
        v[i]=lower_bound(vd+1,vd+tot+1,v[i])-vd;
    }
}
```

适用于下标从1开始的数组



## 莫队算法

### 基础莫队

```c++
typedef long long int ll;
const int N = 50005;
int n,m,maxn,c[N];
struct query {
	int il, ir, id;
	bool operator<(const query &x) const {
		if (il/maxn!=x.il/maxn) return il < x.il;
		return (il/maxn)&1?ir<x.ir:ir>x.ir;
	}
}qs[N];
void work(int co,ll p){...} // 自己设计
// in main
maxn = sqrt(n);
// 输入
sort(qs+1,qs+m+1);
cnt[c[1]]=1; // 初始区间放在了[1,1]，这样不容易错，但是记得要把相关信息也初始化了
for (int i=1,l=1,r=1;i<=m;i++) {
    while (l > qs[i].il) work(c[--l],1);
    while (r < qs[i].ir) work(c[++r],1);
    while (l < qs[i].il) work(c[l++],-1);
    while (r > qs[i].ir) work(c[r--],-1);
}
```



### 带修改莫队（待填坑）



# 数学

## `gcd`

```c++
typedef long long int ll;
ll gcd(ll a, ll b){
	return b==0?a:gcd(b,a%b);
}
```

可以使用$\mathrm{lcm}(a,b)=\frac{ab}{\gcd(a,b)}$求$\mathrm{lcm}$

对于要取模的情况，可以使用分解因数做



## 艾筛

```c++
const int MAXN=1e8+10;
int mnf[MAXN];bool isprime[MAXN];
void init(){
	memset(isprime,true,sizeof(isprime));
	for(int i=1;i<MAXN;++i) mnf[i]=i;
	isprime[0]=isprime[1]=false;
	for(int i=2;i<MAXN;++i){
		if(isprime[i]){
			for(int j=i+i;j<MAXN;++j){
				if(isprime[j]) isprime[j]=false,mnf[j]=i;
			}
		}
	}
}
```

时间复杂度$O(n\log\log n)$



## 带模快速幂

带模版本

```c++
ll qpow(ll a,ll x,ll m){
    ll ret=1;
    while(a){
        if(x&1){
            ret=ret*a%m;
        }
        a=a*a%m;x>>=1;
    }
    return ret;
}
```

不带模版本根据带模版本更改就行



## 快速乘

不要使用$O(\log x)$​的“龟速乘”，除非发现可能被卡精度了，那玩意贼慢

```c++
long long Mul(unsigned long long x,unsigned long long y,ll m){
    unsigned long long tmp=(x*y-(unsigned long long)((long double)x/m*y)*m);
    return (tmp+m)%m;
}
```

（如果非要用龟速乘，直接搬快速幂的板子就行）



## 高维前缀和

对所有的$0\le i\le 2^n-1$，快速求解$\sum\limits_{j\subset i}a_j$

```c++
// f为读入的数据
for(int j = 0; j < n; ++j){
    for(int i = 0; i < 1 << n; ++i){
        if((i >> j) & 1) f[i] += f[i ^ (1 << j)];
    }
}
```

也可以求解$\sum\limits_{i\subset j}a_j$

```c++
for(int j = 0; j < n; ++j){
    for(int i = (1 << n)-1; i >= 0; --i){
        if(((i >> j) & 1)==0) f[i] += f[i ^ (1 << j)];
    }
}
```

时间复杂度$O(n2^n)$​，注意这两个前缀和都包括了自己







# 数据结构

## 并查集

```c++
const int MAXN=1e5+10;
struct USet{
    int fa[MAXN],sz[MAXN],sz;
    void init(){
        sz=_sz;
        for(int i=1;i<=sz;++i){
            fa[i]=i;sz[i]=1;
        }
    }
    int findfa(int x){
        return fa[x]==x?x:fa[x]=findfa(fa[x]);
    }
    bool unio(int x,int y){ // 返回是否合并成功
        int xx=findfa(x),yy=findfa(y);
        if(xx==yy){
            return false;
        }else if(sz[xx]<sz[yy]){
            swap(xx,yy);
        }
        fa[yy]=xx;sz[xx]+=sz[yy];
    }
    int size(int x){
        return sz[findfa(x)];
    }
}uset; // 不要开在栈里。。。有数组
```



## ST表

```c++
const int MAXN=1e5+10,MAXLOG=20;
struct ST{
    static int lg2[MAXN],pow2[MAXN];
    int n,m;int V[MAXN],st[MAXN][MAXLOG];
    static void gen(){
        lg2[1]=0;pow2[0]=1;pow2[1]=2;
        for(int i=2;i<MAXN;++i){
            lg2[i]=lg2[i>>1]+1;pow2[i]=pow2[i-1]<<1;
        }
    }
    void init(){
        for(int i=1;i<=n;++i){
            st[i][0]=V[i];
        }
        for(int j=1;j<MAXLOG;++j){
            for(int i=1;i+pow2[j]-1<=n;++i){
                st[i][j]=max(st[i][j-1],st[i+pow2[j-1]][j-1]);
            }
        }
    }
    int query(int il,int ir){
        int s=lg2[ir-il+1];
        return max(st[il][s],st[ir-pow2[s]+1][s]);
    }
}; // 不要开在栈里，另外如果只有一个ST表就不用写结构体了
// 在main里先ST.gen()，然后再对每个ST表.init()
```



## 动态开点线段树（待填坑）



# 字符串

## 前缀函数（待填坑）



## AC自动机（待填坑）



## 后缀数组（待填坑）



## SAM

```c++
int c2i(char ch){
	return ch-'a';
}
struct SNode{
	int len,link,nxt[26];
};
const int MAXLEN=1e5+10;
struct SAM{
	SNode node[MAXLEN<<1];
	int sz,lst;
	void init(){
		node[0].len=0;node[0].link=-1;sz++;lst=0;
		// 没有清除node的其他信息
	}
	void extend(char ch){
		int cur=sz++,c=c2i(ch);
		node[cur].len=node[lst].len+1; // 新结点p长度一定是lst加一
		int p=lst;
		while(p!=-1 && !node[p].nxt[c]){
			node[p].nxt[c]=cur;p=node[p].link;
			// p的link指p最长非相同endpos后缀的结点
			// 所有的这些结点如果nxt[c]没有的话都应该指向p
		}
		if(p==-1){ // 已经跳到不存在的点了
			node[cur].link=0;
		}else{ // 跳到的点存在
			int q=node[p].nxt[c]; // p通过c转移的结点
			if(node[p].len+1==node[q].len){
				node[cur].link=q;
				// 如果q对应的最长串就是p最长的加上c
				// 那么直接将cur的link给到q就行
			}else{ // 如果不是，就需要克隆点
				int cln=sz++;
				// cln表示的是新的endpos
				// 这个endpos满足：其最长串是p最长串加上c
				node[cln].len=node[p].len+1;
				memcpy(node[cln].nxt,node[q].nxt,sizeof(node[q].nxt));
				node[cln].link=node[q].link;
				while(p!=-1 && node[p].nxt[c]==q){
					node[p].nxt[c]=cln;
					p=node[p].link;
				}
				node[q].link=node[cur].link=cln;
			}
		}
		lst=cur;
	}
};
```

注意：

1. 在构建图的过程中，`link`和`nxt`都是可能改变的，所以不要妄图使用DAG的信息在线计算答案
2. $minlen(v)=len(link(v))+1$​
3. 在SAM中有一条“主链”，也就是整串的链，这条链的结点表示真实存在于原字符串的$endpos$​（或者说右端点），可能有特殊含义
4. `cur`表示的就是右端点所在的`endpos`！利用这点可以求类似“`endpos`中有多少个不同位置的子串”问题
5. 构建完之后，`lst`通过`link`到达的所有状态都是“终止状态”



# 图论

## 最小生成树

注意：判断是否是连通图，下面的板子没有验证连通性

### `prim`

$O(n^2)$​​​的，使用邻接矩阵存图。如果一道题用不了`kruskal`，那就说明是稠密图，那用$O(n^2)$​的`prim`就行

```c++
typedef long long int ll;
const int MAXN=5e3+10;
ll co[MAXN][MAXN],le[MAXN]; // 邻接矩阵 最小值数组
int n;bool vis[MAXN];

ll prim(){
	memset(le,0x3f,sizeof(le));
	ll res=0,len=0;int u=1;
	while(u){
		vis[u]=true;res+=len;
        for(int v=1;v<=n;++v){
            if(!vis[v]){
                le[v]=min(le[v],co[u][v]);
            }
        }
		u=0; // le[0]就是INF
		for(int i=1;i<=n;++i){
			if(!vis[i] && le[i]<le[u]){
				u=i;len=le[i];
			}
		}
	}
	return res;
}
```



### `kruskal`

最常用写法

本质就是`sort`+并查集，就不贴代码了



## 最短路

### `dijkstra`

```c++
const int MAXN=1e5+10;
typedef long long int ll;
const ll INF=0x3f3f3f3f3f3f3f3f;
struct edge{
    int v;ll w;
    edge(int _v=0,ll _w=0):v(_v),w(_w){
    }
};
struct cmpNode{
    bool operator () (const edge & e1,const edge & e2) const {
        return e1.w>e2.w;
    }  
};
vector<edge>G[MAXN];
ll dist[MAXN];int n,m;bool vis[MAXN];
void dijkstra(int s){
    memset(dist,0x3f,sizeof(dist));
    memset(vis,false,sizeof(vis));
    priority_queue<edge,vector<edge>,cmpNode>q;
    dist[s]=0;q.push(edge(s,dist[s]));
    while(q.size()){
        edge e=q.top();q.pop();
        if(vis[e.v]){
            continue;
        }
        vis[e.v]=true;
        for(auto v:G[e.v]){
            if(!vis[v.v] && dist[v.v]>dist[e.v]+v.w){
                dist[v.v]=dist[e.v]+v.w;q.push(edge(v.v,dist[v.v]));
            }
        }
    }
}
```



### `SPFA`

```c++
typedef long long int ll;
struct node {
    int v;
    ll w;
    node(int v=0, ll w=0) : v(v), w(w){};
};
const int MAXN = 1e5+100;
const ll INF = 0x3f3f3f3f3f3f3f3f;
ll dist[MAXN];int dcnt[MAXN];bool inQ[MAXN];
vector<node> G[MAXN];
bool relax(int u,int v,ll w) {
    if (dist[v]>dist[u]+w) {
        dist[v]=dist[u]+w;++dcnt[v];
        return true;
    }else{
        return false;
    }
}
bool SPFA(int s,int cntV) { // 返回是否有负环，s是源，cntV是点的数量
    memset(dist, (int)INF, sizeof(dist));memset(dcnt, 0, sizeof(dcnt));
    queue<int>Q;inQ[s]=true;dist[s] = 0;Q.push(s);
    while (!Q.empty()) {
        int u = Q.front();Q.pop();
        inQ[u]=false;
        for (node i : G[u]) {
            if ( relax(u, i.v, i.w) && !inQ[i.v] ) {
                if (dcnt[i.v]>=cntV+1) {
                    return true;
                }
                Q.push(v);
                inQ[v]=true;
            }
        }
    }
    return false;
}
```



### `floyd`

```c++
typedef long long int ll;
const ll INF=0x3f3f3f3f3f3f3f3f;
const int MAXN=3e2+10;
int n;
ll dist[MAXN][MAXN];
void floyd(){
    // 先将dist里面图的边权标上，其他的是INF
    for (int k=1; k<=n; k++) {
        for (int i=1; i<=n; i++) {
            for (int j=1; j<=n; j++) {
                dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j]);
            }
        }
    }
}
```



## 拓扑排序

注意在需要字典序最小的情况可以使用优先队列维护待遍历队列，其他的不写了



## 有向图欧拉回路/路径

```c++
const int MAXN=1e5+10;
vector<int>G[MAXN];
int en[MAXN],n,m,din[MAXN],dout[MAXN];
void add(int u,int v){
	G[u].push_back(v);++din[v];++dout[u];
}
stack<int>res;
void dfs(int u){
	int ie=en[u]++;
	while(ie<G[u].size()){
		dfs(G[u][ie]);
		ie=en[u]++;
	}
	res.push(u);
}
// in main
int s=1; // 起点，注意需要先判断有没有欧拉回路/路径存在，以及起点应该在哪
dfs(s);
while(res.size()){
    cout<<res.top()<<' ';res.pop();
}
```



## 强连通分量

```c++
// Tarjan算法，注意这里dfn[0]和scid[0]都表示计数器
int dfn[MAXN],low[MAXN],scid[MAXN];
struct SC{
	int sz,pt;
}sc[MAXN];
stack<int>scst;
bool inst[MAXN];
void scdfs(int u){
	dfn[u]=low[u]=++dfn[0];
	scst.push(u);inst[u]=true;
	for(auto v:G[u]){
		if(!dfn[v]){
			scdfs(v);low[u]=min(low[u],low[v]);
		}else if(inst[v]){
			low[u]=min(low[u],dfn[v]);
			// 这里写dfn[v]和low[v]是一样的
		}
	}
	if(dfn[u]==low[u]){
		int id=++scid[0],v;
		while(scst.top()!=u){
			v=scst.top();scst.pop();inst[v]=false;
			++sc[id].sz;scid[v]=id;
		}
		scst.pop();inst[u]=false;
		++sc[id].sz;scid[u]=id;
	}
}
```



## SAT问题

### Horn-SAT

数据格式

1. 命题`x`为真：`x`
2. 命题`x`为假：`!x`
3. 命题`a`、`b`等可以推出`x`为真：`a b -> x`
4. 命题`a`、`b`等可以推出`x`为假：`a b -> !x`

其中`a`、`b`、`x`均为数字

```c++
const int MAXN=1e6+10;
int n,m,res[MAXN],cnt[MAXN],c2a[MAXN];
// n:陈述句个数 m:命题个数
// res[i]表示i这个命题是否被确定
// cnt[c]表示c这个条件还有几个前件没有确定
// c2a[c]表示c这个条件得到的结论
vector<int>a2c[MAXN]; // a2c[a]表示a这个前件会指向的集合
bool usable[MAXN]; // usable[c]表示c这个条件集合已经没用了(存在前件为假)

void init(){
	memset(res,-1,sizeof(res));memset(usable,true,sizeof(usable));
}
void noAns(){
	cout<<"conflict\n";exit(0);
}
typedef char * pchar;
int isNum(pchar & pc){
	int re=0;
	while(*pc && *pc!=' ' && *pc!='\n'){
		if(*pc<'0' || *pc>'9'){
			++pc;
		}else{
			re=re*10+(*pc++)-'0';
		}
	}
	++pc;
	return re;
}
queue<int>aq;
bool assign(int a,int x){
	if(res[a]==!x){
		noAns();
		return false;
	}else if(res[a]==x){
		return false;
	}else{
		res[a]=x;
		return true;
	}
}
void gen(char * str,int c){
	char * p=str;
	if(strstr(str,"->")!=nullptr){
		int a;
		while( (a=isNum(p)) ){
			a2c[a].push_back(c);++cnt[c];
		}
		if(*p=='!'){ // a<0,为...->!a
			++p;a=-isNum(p);
		}else{
			a=isNum(p);
		}
		c2a[c]=a;
	}else{
		int a;
		if(*p=='!'){
			++p;a=isNum(p);assign(a,0);
		}else{
			a=isNum(p);assign(a,1);
		}
		aq.push(a);
	}
}
const int MAXLEN=MAXN<<2;
char str[MAXLEN];

void work(){
	while(aq.size()){
		int a=aq.front();aq.pop();
		if(res[a]==0){
			for(int c:a2c[a]){
				usable[c]=false;
			}
		}
		for(int c:a2c[a]){
			--cnt[c];
			if(cnt[c]==0 && usable[c]){
				usable[c]=false;
				int _a=c2a[c];
				if(_a<0){
					if(assign(-_a,0)){
						aq.push(-_a);
					}
				}else if(_a>0){
					if(assign(_a,1)){
						aq.push(_a);
					}
				}
			}
		}
	}
}

void prt(){
	for(int i=1;i<=m;++i){
		printf(res[i]==1?"T":"F");
	}
	printf("\n");
}
```

先`init()`，输入语句`str`后放入`gen(str)`里，然后跑`work()`就行



### 字典序最小2-SAT

```c++
const int MAXN=1e5+10;
struct TwoSatBF{ // 暴力求解字典序最小的解
	int n;vector<int>G[MAXN<<1];
	bool slt[MAXN<<1];
	// 偶数点：false 奇数点：true 这样x^1就是反面
	void init(int _n){
		n=_n;
		for(int i=0;i<(n<<1);++i){
			G[i].clear();slt[i]=false;
		}
	}
	void addLimit(int x,int y){
		// 选了x就要选y，具体看情况使用
		G[x].push_back(y);
		G[y^1].push_back(x^1);
	}
	stack<int>st;
	void clearst(){while(st.size()) st.pop();}
	bool dfs(int u){
		if(slt[u^1]){
			return false;
		}else if(slt[u]){
			return true;
		}
		slt[u]=true;
		st.push(u);
		for(auto v:G[u]){
			if(!dfs(v)){
				return false;
			}
		}
		return true;
	}
	bool solve(){
		for(int u=0;u<(n<<1);u+=2){
			if(!slt[u] && !slt[u^1]){
				clearst();
				if(!dfs(u)){
					clearst();
					if(!dfs(u^1)){
						return fales;
					}
				}
			}
		}
		return true;
	}
};
```



### $O(n+m)$的2-SAT

```c++
const int MAXN=2e6+10; // 注意开两倍空间
struct TwoSatSC{
	// x和x+1一组，其中x=2k，共n组，编号[0,2n-1]，注意编号是从0开始！
	void init(int _n){ // 多样例记得memset
		n=_n;
	}
	vector<int>G[MAXN];
	void add(int u,int v){
		// 选了u就要选v，不自带对称建边
		G[u].push_back(v);
	}
	int n,dfn[MAXN],low[MAXN],scid[MAXN];
	struct SC{
		int sz,pt;
	}sc[MAXN];
	stack<int>scst;
	bool inst[MAXN];
	void scdfs(int u){
		dfn[u]=low[u]=++dfn[0];
		scst.push(u);inst[u]=true;
		for(auto v:G[u]){
			if(!dfn[v]){
				scdfs(v);low[u]=min(low[u],low[v]);
			}else if(inst[v]){
				low[u]=min(low[u],dfn[v]);
				// 这里dfn[v]和low[v]应该一样的233
			}
		}
		if(dfn[u]==low[u]){
			int id=++scid[0],v;
			while(scst.top()!=u){
				v=scst.top();scst.pop();inst[v]=false;
				++sc[id].sz;scid[v]=id;
			}
			scst.pop();inst[u]=false;
			++sc[id].sz;scid[u]=id;
		}
	}
	bool check(){
		for(int i=0;i<2*n;++i){
			if(!dfn[i]){
				scdfs(i);
			}
		}
		for(int i=0;i<2*n;i+=2){
			if(scid[i]==scid[i+1]){
				return false;
			}
		}
		return true;
	}
	void prt(){
		for(int i=0;i<2*n;i+=2){
			cout<<(scid[i]<scid[i+1]?"0 ":"1 ");
		}
	}
};
```



## 点双与边双、圆方树（待填坑）





# 网络流

## 二分图最大权匹配-匈牙利算法

时间复杂度：$O(n^3)$

```c++
template <typename T>struct hungarian {
	int n;
	vector<int> matchx,matchy;  // 左右集合对应的匹配点
	vector<int> pre;     // 连接右集合的左点
	vector<bool> visx,visy;   // 拜访数组 左右
	vector<T> lx,ly;
	vector<vector<T> > g;
	vector<T> slack;
	T inf,res;
	queue<int> q;
	int org_n,org_m;

	hungarian(int _n, int _m) {
		org_n = _n;
		org_m = _m;
		n = max(_n, _m);
		inf = numeric_limits<T>::max();
		res = 0;
		g = vector<vector<T> >(n, vector<T>(n));
		matchx = vector<int>(n, -1);
		matchy = vector<int>(n, -1);
		pre = vector<int>(n);
		visx = vector<bool>(n);
		visy = vector<bool>(n);
		lx = vector<T>(n, -inf);
		ly = vector<T>(n);
		slack = vector<T>(n);
	}

	void addEdge(int u, int v, T w) {
        if(w<0){ // 负权不如不匹配
            g[u][v]=0;
        }else{
            g[u][v] = w;
        }
	}
	bool check(int v) {
		visy[v] = true;
		if (matchy[v] != -1) {
			q.push(matchy[v]);
			visx[matchy[v]] = true;  // in S
			return false;
		}
		// 找到新的未匹配点 更新匹配点 pre 数组记录着"非匹配边"上与之相连的点
		while (v != -1) {
			matchy[v] = pre[v];
			swap(v, matchx[pre[v]]);
		}
		return true;
	}

	void bfs(int i) {
	    while (!q.empty()) {
		    q.pop();
	    }
	    q.push(i);
		visx[i] = true;
		while (true) {
			while (!q.empty()) {
				int u = q.front();
				q.pop();
				for (int v = 0; v < n; v++) {
					if (!visy[v]) {
						T delta = lx[u] + ly[v] - g[u][v];
						if (slack[v] >= delta) {
							pre[v] = u;
							if (delta) {
		            			slack[v] = delta;
		          			} else if (check(v)) {  // delta=0 代表有机会加入相等子图 找增广路
		                                  // 找到就return 重建交错树
		            			return;
		          			}
		        		}
		      		}
		    	}
		  	}
		  // 没有增广路 修改顶标
		  	T a = inf;
			for (int j = 0; j < n; j++) {
			    if (!visy[j]) {
				    a = min(a, slack[j]);
		    	}
		  	}
		  	for (int j = 0; j < n; j++) {
		    	if (visx[j]) {  // S
		    		lx[j] -= a;
		    	}
		    	if (visy[j]) {  // T
		    		ly[j] += a;
		    	} else {  // T'
		    		slack[j] -= a;
		    	}
		  	}
		  	for (int j = 0; j < n; j++) {
		    	if (!visy[j] && slack[j] == 0 && check(j)) {
		      		return;
		    	}
		  	}
		}
	}

	void solve() {
		// 初始顶标
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				lx[i] = max(lx[i], g[i][j]);
			}
    	}

    	for (int i = 0; i < n; i++) {
    		fill(slack.begin(), slack.end(), inf);
    		fill(visx.begin(), visx.end(), false);
    		fill(visy.begin(), visy.end(), false);
    		bfs(i);
    	}

    	// custom
    	for (int i = 0; i < n; i++) {
    		if (g[i][matchx[i]] > 0) {
        		res += g[i][matchx[i]];
      		} else {
        		matchx[i] = -1;
      		}
    	}
	    // cout << res << "\n";
	    // for (int i = 0; i < org_n; i++) {
	    // 	cout << matchx[i] + 1 << " ";
	    // }
	    // cout << "\n";
  	}
};
// in main.cpp
int n;
cin>>n>>m; // 图左右的点数量
hungarian solver(n,m);
while(ecnt--){
    int u,v,w;
    solver.addEdge(u-1,v-1,w); // solver里面的下标都是从0开始，u,v表示左右第几个点
}
```

