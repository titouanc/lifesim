// A spatial query to lookup in the tree
#[derive(Copy, Clone)]
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
}

// See https://aloso.github.io/2021/03/09/creating-an-iterator
pub struct NodeIter<'a, K, V> {
    parent: Option<Box<NodeIter<'a, K, V>>>,
    started: bool,
    to_visit: Vec<&'a Node<K, V>>,
    query: Query<K>,
}

impl<K, V> NodeIter<'_, K, V> {
    fn depth(&self, acc: usize) -> usize {
        match &self.parent {
            None => acc,
            Some(x) => x.depth(1 + acc),
        }
    }
}

impl<'a, K, V> NodeIter<'a, K, V> {
    fn new(parent: Option<Box<NodeIter<'a, K, V>>>, node: &'a Node<K, V>, query: Query<K>) -> NodeIter<'a, K, V> {
        let mut to_visit: Vec<&'a Node<K, V>> = vec![];
        for child in &node.children {
            match child {
                Some(x) => to_visit.push(&x),
                _ => {}
            }
        }
        to_visit.push(node);

        NodeIter {
            started: false,
            parent: parent,
            to_visit: to_visit,
            query: query
        }
    }
}

impl<K, V> Default for NodeIter<'_, K, V> {
    fn default() -> Self {
        NodeIter {
            parent: None,
            started: false,
            to_visit: vec![],
            query: Query::Everything,
        }
    }
}

impl<'a, K: Copy, V> Iterator for NodeIter<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        use std::mem;

        match self.to_visit.pop() {
            None => match self.parent.take() {
                None => None,
                Some(parent) => {
                    *self = *parent;
                    self.next()
                }
            },
            Some(node) => {
                if self.started {
                    *self = NodeIter::new(Some(Box::new(mem::take(self))), node, self.query);
                    self.next()
                } else {
                    self.started = true;
                    Some(&node.value)
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

    fn child_index(&self, x: K, y: K) -> usize {
        let qx = (x < self.x) as usize;
        let qy = (y < self.y) as usize;
        2 * qx + qy
    }

    fn insert(&mut self, x: K, y: K, value: V) {
        let idx = self.child_index(x, y);
        match &mut self.children[idx] {
            Some(node) => node.insert(x, y, value),
            None => self.children[idx] = Some(Box::new(Node::new(x, y, value))),
        }
    }
}

pub struct QuadTree<K, V> {
    root: Option<Node<K, V>>,
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

    fn iter_from_root(&self, query: Query<K>) -> NodeIter<'_, K, V> {
        match &self.root {
            Some(node) => NodeIter::new(None, node, query),
            None => NodeIter::default(),
        }
    }

    pub fn iter(&self) -> NodeIter<'_, K, V> {
        self.iter_from_root(Query::Everything)
    }

    pub fn rect(&self, x: K, y: K, w: K, h: K) -> NodeIter<'_, K, V> {
        self.iter_from_root(Query::Rect {x:x, y:y, w:w, h:h})
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
        let v1: Vec<i32> = q.iter().copied().collect();
        assert!(v1.len() == 0);

        populate_qtree(&mut q);

        let v2: Vec<i32> = q.iter().copied().collect();
        assert!(v2.len() == 5);
    }

    #[test]
    fn iter_on_rect() {
        let mut q: QuadTree<f64, i32> = QuadTree::new();
        populate_qtree(&mut q);

        let v2: Vec<i32> = q.rect(0., 0., 1., 1.).copied().collect();
        assert!(v2.len() == 2);
    }
}