import { openDB, IDBPDatabase } from "idb";

let db: IDBPDatabase;

async function getDB() {
  if (!db) {
    db = await openDB("db1", 1, {
      upgrade(db) {
        db.createObjectStore("paths", { keyPath: "path" });
      },
      blocked(currentVersion, blockedVersion, event) {},
      blocking(currentVersion, blockedVersion, event) {},
      terminated() {},
    });
  }
  return db;
}

export default getDB;

export async function insert(storeName: string, value: string) {
  const db = await getDB();
  await db.add(storeName, value);
}

export async function findAll(storeName: string) {
  const db = await getDB();
  await db.getAll(storeName);
}

export async function edit(storeName: string, value: string) {
  const db = await getDB();
  await db.put(storeName, value);
}
