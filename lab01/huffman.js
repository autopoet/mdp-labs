/**
 * 霍夫曼编码实验
 */
const targetStr = 'helloworld';

// 1. 统计频率并初始化
const map = new Map();
[...targetStr].forEach((c, i) => {
  map.has(c) ? map.get(c).freq++ : map.set(c, { char: c, freq: 1, order: i, left: null, right: null });
});

// 排序：频率升序 -> 出现顺序
const queue = [...map.values()].sort((a, b) => a.freq - b.freq || a.order - b.order);

// 2. 建树 (每次取前两个，合成后直接塞末尾)
while (queue.length > 1) {
  const left = queue.shift();
  const right = queue.shift();
  queue.push({ char: null, freq: left.freq + right.freq, left, right });
}
const root = queue[0];

// 3. 生成编码字典表 
const codeMap = {};
const getCodes = (node, code = "") => {
  if (node.char) return codeMap[node.char] = code;
  getCodes(node.left, code + "0");
  getCodes(node.right, code + "1");
};
getCodes(root);

// 4. 编解码与验证
const encoded = [...targetStr].map(c => codeMap[c]).join('');

let decoded = "", curr = root;
for (const bit of encoded) {
  curr = bit === '0' ? curr.left : curr.right;
  if (curr.char) {
    decoded += curr.char;
    curr = root;
  }
}

// === 输出 ===
console.log("1. 编码字典:", codeMap);
console.log(`2. 编码结果: ${encoded}`);
console.log(`3. 解码结果: ${decoded}`);
console.log(` 验证: ${decoded === targetStr ? "成功" : "失败"}`);


// 这里我使用了 forEach 遍历字符串数组，通过一个回调函数来维护外部定义的 Map 对象。
// 在回调函数内部，我利用三元运算符实现了逻辑判断：如果字符已存在，则通过 get 获取引用并累加频率；如果不存在，则通过 set 初始化该字符的属性（包括频率、首次出现顺序以及后续建树所需的指针）。

