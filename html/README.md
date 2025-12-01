To sync only metadata:

```
rsync -a -R --prune-empty-dirs --include '*/' --include '*metadata*.hdf' --exclude '*' "<user>>@</host>>:/home/<user>>/dasegen/datasets/" "./dasegen/datasets"
```