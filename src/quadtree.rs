const BOTTOM_LEFT: usize = 0;
const TOP_LEFT: usize = 1;
const BOTTOM_RIGHT: usize = 2;
const TOP_RIGHT: usize = 3;

// A spatial query to lookup in the tree
enum Query<K> {
    Everything,
    Rect {x: K, y: K, w: K, h: K},
}

// Units of storage within the tree
struct Node<K, V> {
    children: [Option<Box<Node<K, V>>>; 4],
    value: V,
    x: K,
    y: K,
}

impl<K: std::cmp::PartialOrd + std::ops::Add<Output=K> + Copy> Query<K> {
    fn contains<V>(&self, node: &Node<K, V>) -> bool {
        match self {
            &Query::Everything =>
                true,
            &Query::Rect {x, y, w, h} =>
                x < node.x && node.x < x + w && y < node.y && node.y < y + h,
        }
    }

    fn covers<V>(&self, node: &Node<K, V>, quadrant: usize) -> bool {
        match self {
            &Query::Everything => true,
            &Query::Rect {x, y, w, h} => {
                match quadrant {
                    BOTTOM_LEFT => x < node.x && y < node.y,
                    TOP_LEFT => x < node.x && y + h > node.y,
                    BOTTOM_RIGHT => x + w > node.x && y < node.y,
                    TOP_RIGHT => x + w > node.x && y + w > node.y,
                    _ => false,
                }
            }
        }
    }
}

impl<K: std::cmp::PartialOrd + std::ops::Add<Output = K> + Copy, V> Node<K, V> {
    fn new(x: K, y: K, value: V) -> Node<K, V> {
        Node {
            children: [None, None, None, None],
            value: value,
            x: x,
            y: y
        }
    }

    fn quadrant_index(&self, x: K, y: K) -> usize {
        let qx = (x < self.x) as usize;
        let qy = (y < self.y) as usize;
        2 * qx + qy
    }

    fn insert(&mut self, x: K, y: K, value: V) {
        let idx = self.quadrant_index(x, y);
        match &mut self.children[idx] {
            Some(node) => node.insert(x, y, value),
            None => self.children[idx] = Some(Box::new(Node::new(x, y, value))),
        }
    }
}

pub struct QuadTree<K, V> {
    root: Option<Node<K, V>>,
}

struct NodeScanner<'a, K, V> {
    children_seen: usize,
    node: &'a Node<K, V>,
}

pub struct QuadTreeIter<'a, K, V> {
    stack: Vec<NodeScanner<'a, K, V>>,
    query: Query<K>,
}

impl<K, V> Default for QuadTreeIter<'_, K, V> {
    fn default() -> Self {
        QuadTreeIter {
            stack: vec![],
            query: Query::Everything,
        }
    }
}

impl<'a, K, V> QuadTreeIter<'a, K, V> {
    fn new(root: &'a Node<K, V>, query: Query<K>) -> QuadTreeIter<'a, K, V> {
        QuadTreeIter {
            stack: vec![NodeScanner {node: root, children_seen: 0}],
            query: query,
        }
    }
}

impl<'a, K: std::cmp::PartialOrd + std::ops::Add<Output=K> + Copy, V> Iterator for QuadTreeIter<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stack.len() == 0 {
            return None;
        }
        
        let top_index = self.stack.len() - 1;
        let scanner = &mut self.stack[top_index];

        if scanner.children_seen >= 4 {
            let removed = self.stack.pop().unwrap();
            if self.query.contains(removed.node) {
                Some(&removed.node.value)
            } else {
                self.next()
            }
        } else {
            for i in scanner.children_seen..4 {
                let idx = scanner.children_seen;
                scanner.children_seen += 1;

                if ! self.query.covers(scanner.node, idx){
                    continue;
                }

                match &scanner.node.children[idx] {
                    Some(node) => {
                        self.stack.push(NodeScanner {
                            children_seen: 0,
                            node: &node,
                        });
                        break;
                    }
                    _ => {}
                }
            }
            self.next()
        }
    }
}

impl<'a, K: std::cmp::PartialOrd + std::ops::Add<Output = K> + Copy, V> QuadTree<K, V> {
    pub fn new() -> QuadTree<K, V> {
        QuadTree {root: None}
    }

    pub fn insert(&mut self, x: K, y: K, value: V) {
        match &mut self.root {
            Some(node) => node.insert(x, y, value),
            None => self.root = Some(Node::new(x, y, value)),
        }
    }

    fn iter_from_root(&self, query: Query<K>) -> QuadTreeIter<'_, K, V> {
        match &self.root {
            Some(node) => QuadTreeIter::new(node, query),
            None => QuadTreeIter::default(),
        }
    }

    pub fn iter(&self) -> QuadTreeIter<'_, K, V> {
        self.iter_from_root(Query::Everything)
    }

    pub fn rect(&self, x: K, y: K, w: K, h: K) -> QuadTreeIter<'_, K, V> {
        self.iter_from_root(Query::Rect {x:x, y:y, w:w, h:h})
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn populate_qtree(tree: &mut QuadTree<f64, i32>) {
        tree.insert(0.250, 0.250, 1);
        tree.insert(0.500, 0.500, 2);
        tree.insert(1.500, 1.500, 3);
        tree.insert(0.500, 1.500, 4);
        tree.insert(1.500, 0.500, 5);
    }

    #[test]
    fn iter_on_everything() {
        let mut q: QuadTree<f64, i32> = QuadTree::new();
        let v1: HashSet<i32> = q.iter().copied().collect();
        assert!(v1.len() == 0);

        populate_qtree(&mut q);

        let v2: HashSet<i32> = q.iter().copied().collect();
        assert!(v2.len() == 5);
    }

    #[test]
    fn iter_on_rect() {
        let mut q: QuadTree<f64, i32> = QuadTree::new();
        populate_qtree(&mut q);

        let v2: HashSet<i32> = q.rect(0., 0., 1., 1.).copied().collect();
        assert!(v2.len() == 2);
    }
}