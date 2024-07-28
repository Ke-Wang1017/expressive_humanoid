/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




class sipFbxLayerElementArrayTemplate_FbxVector2 : public  ::FbxLayerElementArrayTemplate_FbxVector2
{
public:
    sipFbxLayerElementArrayTemplate_FbxVector2( ::EFbxType);
    sipFbxLayerElementArrayTemplate_FbxVector2(const  ::FbxLayerElementArrayTemplate_FbxVector2&);
    virtual ~sipFbxLayerElementArrayTemplate_FbxVector2();

    /*
     * There is a protected method for every virtual method visible from
     * this class.
     */
protected:
    void* GetLocked( ::FbxLayerElementArray::ELockMode, ::EFbxType);
    void Release(void**, ::EFbxType);
     ::size_t GetStride() const;

public:
    sipSimpleWrapper *sipPySelf;

private:
    sipFbxLayerElementArrayTemplate_FbxVector2(const sipFbxLayerElementArrayTemplate_FbxVector2 &);
    sipFbxLayerElementArrayTemplate_FbxVector2 &operator = (const sipFbxLayerElementArrayTemplate_FbxVector2 &);

    char sipPyMethods[3];
};

sipFbxLayerElementArrayTemplate_FbxVector2::sipFbxLayerElementArrayTemplate_FbxVector2( ::EFbxType a0):  ::FbxLayerElementArrayTemplate_FbxVector2(a0), sipPySelf(0)
{
    memset(sipPyMethods, 0, sizeof (sipPyMethods));
}

sipFbxLayerElementArrayTemplate_FbxVector2::sipFbxLayerElementArrayTemplate_FbxVector2(const  ::FbxLayerElementArrayTemplate_FbxVector2& a0):  ::FbxLayerElementArrayTemplate_FbxVector2(a0), sipPySelf(0)
{
    memset(sipPyMethods, 0, sizeof (sipPyMethods));
}

sipFbxLayerElementArrayTemplate_FbxVector2::~sipFbxLayerElementArrayTemplate_FbxVector2()
{
    sipInstanceDestroyed(sipPySelf);
}

void* sipFbxLayerElementArrayTemplate_FbxVector2::GetLocked( ::FbxLayerElementArray::ELockMode a0, ::EFbxType a1)
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[0],sipPySelf,NULL,sipName_GetLocked);

    if (!sipMeth)
        return  ::FbxLayerElementArrayTemplate_FbxVector2::GetLocked(a0,a1);

    extern void* sipVH_fbx_7(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *,  ::FbxLayerElementArray::ELockMode, ::EFbxType);

    return sipVH_fbx_7(sipGILState, 0, sipPySelf, sipMeth, a0, a1);
}

void sipFbxLayerElementArrayTemplate_FbxVector2::Release(void**a0, ::EFbxType a1)
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,&sipPyMethods[1],sipPySelf,NULL,sipName_Release);

    if (!sipMeth)
    {
         ::FbxLayerElementArrayTemplate_FbxVector2::Release(a0,a1);
        return;
    }

    extern void sipVH_fbx_8(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *, void**, ::EFbxType);

    sipVH_fbx_8(sipGILState, 0, sipPySelf, sipMeth, a0, a1);
}

 ::size_t sipFbxLayerElementArrayTemplate_FbxVector2::GetStride() const
{
    sip_gilstate_t sipGILState;
    PyObject *sipMeth;

    sipMeth = sipIsPyMethod(&sipGILState,const_cast<char *>(&sipPyMethods[2]),sipPySelf,NULL,sipName_GetStride);

    if (!sipMeth)
        return  ::FbxLayerElementArrayTemplate_FbxVector2::GetStride();

    extern  ::size_t sipVH_fbx_9(sip_gilstate_t, sipVirtErrorHandlerFunc, sipSimpleWrapper *, PyObject *);

    return sipVH_fbx_9(sipGILState, 0, sipPySelf, sipMeth);
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_Add, "Add(self, FbxVector2) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_Add(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_Add(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxVector2* a0;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, sipType_FbxVector2, &a0))
        {
            int sipRes;

            sipRes = sipCpp->Add(*a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_Add, doc_FbxLayerElementArrayTemplate_FbxVector2_Add);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_InsertAt, "InsertAt(self, int, FbxVector2) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_InsertAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_InsertAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxVector2* a1;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, &a0, sipType_FbxVector2, &a1))
        {
            int sipRes;

            sipRes = sipCpp->InsertAt(a0,*a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_InsertAt, doc_FbxLayerElementArrayTemplate_FbxVector2_InsertAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_SetAt, "SetAt(self, int, FbxVector2)");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_SetAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_SetAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxVector2* a1;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, &a0, sipType_FbxVector2, &a1))
        {
            sipCpp->SetAt(a0,*a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_SetAt, doc_FbxLayerElementArrayTemplate_FbxVector2_SetAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_SetLast, "SetLast(self, FbxVector2)");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_SetLast(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_SetLast(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxVector2* a0;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, sipType_FbxVector2, &a0))
        {
            sipCpp->SetLast(*a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_SetLast, doc_FbxLayerElementArrayTemplate_FbxVector2_SetLast);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveAt, "RemoveAt(self, int) -> FbxVector2");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, &a0))
        {
             ::FbxVector2*sipRes;

            sipRes = new  ::FbxVector2(sipCpp->RemoveAt(a0));

            return sipConvertFromNewType(sipRes,sipType_FbxVector2,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_RemoveAt, doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveLast, "RemoveLast(self) -> FbxVector2");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveLast(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveLast(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp))
        {
             ::FbxVector2*sipRes;

            sipRes = new  ::FbxVector2(sipCpp->RemoveLast());

            return sipConvertFromNewType(sipRes,sipType_FbxVector2,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_RemoveLast, doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveLast);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveIt, "RemoveIt(self, FbxVector2) -> bool");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveIt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveIt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxVector2* a0;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, sipType_FbxVector2, &a0))
        {
            bool sipRes;

            sipRes = sipCpp->RemoveIt(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_RemoveIt, doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveIt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_GetAt, "GetAt(self, int) -> FbxVector2");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_GetAt(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_GetAt(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
        const  ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "Bi", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, &a0))
        {
             ::FbxVector2*sipRes;

            sipRes = new  ::FbxVector2(sipCpp->GetAt(a0));

            return sipConvertFromNewType(sipRes,sipType_FbxVector2,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_GetAt, doc_FbxLayerElementArrayTemplate_FbxVector2_GetAt);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_GetFirst, "GetFirst(self) -> FbxVector2");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_GetFirst(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_GetFirst(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp))
        {
             ::FbxVector2*sipRes;

            sipRes = new  ::FbxVector2(sipCpp->GetFirst());

            return sipConvertFromNewType(sipRes,sipType_FbxVector2,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_GetFirst, doc_FbxLayerElementArrayTemplate_FbxVector2_GetFirst);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_GetLast, "GetLast(self) -> FbxVector2");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_GetLast(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_GetLast(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp))
        {
             ::FbxVector2*sipRes;

            sipRes = new  ::FbxVector2(sipCpp->GetLast());

            return sipConvertFromNewType(sipRes,sipType_FbxVector2,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_GetLast, doc_FbxLayerElementArrayTemplate_FbxVector2_GetLast);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_Find, "Find(self, FbxVector2) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_Find(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_Find(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxVector2* a0;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, sipType_FbxVector2, &a0))
        {
            int sipRes;

            sipRes = sipCpp->Find(*a0);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_Find, doc_FbxLayerElementArrayTemplate_FbxVector2_Find);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_FindAfter, "FindAfter(self, int, FbxVector2) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_FindAfter(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_FindAfter(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxVector2* a1;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, &a0, sipType_FbxVector2, &a1))
        {
            int sipRes;

            sipRes = sipCpp->FindAfter(a0,*a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_FindAfter, doc_FbxLayerElementArrayTemplate_FbxVector2_FindAfter);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2_FindBefore, "FindBefore(self, int, FbxVector2) -> int");

extern "C" {static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_FindBefore(PyObject *, PyObject *);}
static PyObject *meth_FbxLayerElementArrayTemplate_FbxVector2_FindBefore(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        int a0;
         ::FbxVector2* a1;
         ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BiJ9", &sipSelf, sipType_FbxLayerElementArrayTemplate_FbxVector2, &sipCpp, &a0, sipType_FbxVector2, &a1))
        {
            int sipRes;

            sipRes = sipCpp->FindBefore(a0,*a1);

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName_FindBefore, doc_FbxLayerElementArrayTemplate_FbxVector2_FindBefore);

    return NULL;
}


extern "C" {static PyObject *slot_FbxLayerElementArrayTemplate_FbxVector2___getitem__(PyObject *,PyObject *);}
static PyObject *slot_FbxLayerElementArrayTemplate_FbxVector2___getitem__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp = reinterpret_cast< ::FbxLayerElementArrayTemplate_FbxVector2 *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxLayerElementArrayTemplate_FbxVector2));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        int a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1i", &a0))
        {
             ::FbxVector2*sipRes = 0;
            int sipIsErr = 0;

#line 290 "/home/ke/Documents/expressive-humanoid/sip/fbxlayerelementarray.sip"
        if (a0 < 0 || a0 >= sipCpp->GetCount())
        {
            PyErr_Format(PyExc_IndexError, "sequence index out of range");
            sipIsErr = 1;
        }
        else
        {
            fbxArrayElementCopy(&sipRes, (FbxVector2*)NULL, sipCpp, a0);
        }
#line 538 "/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipfbxFbxLayerElementArrayTemplate_FbxVector2.cpp"

            if (sipIsErr)
                return 0;

            return sipConvertFromNewType(sipRes,sipType_FbxVector2,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayerElementArrayTemplate_FbxVector2, sipName___getitem__, NULL);

    return 0;
}


/* Cast a pointer to a type somewhere in its inheritance hierarchy. */
extern "C" {static void *cast_FbxLayerElementArrayTemplate_FbxVector2(void *, const sipTypeDef *);}
static void *cast_FbxLayerElementArrayTemplate_FbxVector2(void *sipCppV, const sipTypeDef *targetType)
{
     ::FbxLayerElementArrayTemplate_FbxVector2 *sipCpp = reinterpret_cast< ::FbxLayerElementArrayTemplate_FbxVector2 *>(sipCppV);

    if (targetType == sipType_FbxLayerElementArray)
        return static_cast< ::FbxLayerElementArray *>(sipCpp);

    return sipCppV;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxLayerElementArrayTemplate_FbxVector2(void *, int);}
static void release_FbxLayerElementArrayTemplate_FbxVector2(void *sipCppV, int sipState)
{
    if (sipState & SIP_DERIVED_CLASS)
        delete reinterpret_cast<sipFbxLayerElementArrayTemplate_FbxVector2 *>(sipCppV);
    else
        delete reinterpret_cast< ::FbxLayerElementArrayTemplate_FbxVector2 *>(sipCppV);
}


extern "C" {static void dealloc_FbxLayerElementArrayTemplate_FbxVector2(sipSimpleWrapper *);}
static void dealloc_FbxLayerElementArrayTemplate_FbxVector2(sipSimpleWrapper *sipSelf)
{
    if (sipIsDerivedClass(sipSelf))
        reinterpret_cast<sipFbxLayerElementArrayTemplate_FbxVector2 *>(sipGetAddress(sipSelf))->sipPySelf = NULL;

    if (sipIsOwnedByPython(sipSelf))
    {
        release_FbxLayerElementArrayTemplate_FbxVector2(sipGetAddress(sipSelf), sipIsDerivedClass(sipSelf));
    }
}


extern "C" {static void *init_type_FbxLayerElementArrayTemplate_FbxVector2(sipSimpleWrapper *, PyObject *, PyObject *, PyObject **, PyObject **, PyObject **);}
static void *init_type_FbxLayerElementArrayTemplate_FbxVector2(sipSimpleWrapper *sipSelf, PyObject *sipArgs, PyObject *sipKwds, PyObject **sipUnused, PyObject **, PyObject **sipParseErr)
{
    sipFbxLayerElementArrayTemplate_FbxVector2 *sipCpp = 0;

    {
         ::EFbxType a0;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "E", sipType_EFbxType, &a0))
        {
            sipCpp = new sipFbxLayerElementArrayTemplate_FbxVector2(a0);

            sipCpp->sipPySelf = sipSelf;

            return sipCpp;
        }
    }

    {
        const  ::FbxLayerElementArrayTemplate_FbxVector2* a0;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "J9", sipType_FbxLayerElementArrayTemplate_FbxVector2, &a0))
        {
            sipCpp = new sipFbxLayerElementArrayTemplate_FbxVector2(*a0);

            sipCpp->sipPySelf = sipSelf;

            return sipCpp;
        }
    }

    return NULL;
}


/* Define this type's super-types. */
static sipEncodedTypeDef supers_FbxLayerElementArrayTemplate_FbxVector2[] = {{176, 255, 1}};


/* Define this type's Python slots. */
static sipPySlotDef slots_FbxLayerElementArrayTemplate_FbxVector2[] = {
    {(void *)slot_FbxLayerElementArrayTemplate_FbxVector2___getitem__, getitem_slot},
    {0, (sipPySlotType)0}
};


static PyMethodDef methods_FbxLayerElementArrayTemplate_FbxVector2[] = {
    {SIP_MLNAME_CAST(sipName_Add), meth_FbxLayerElementArrayTemplate_FbxVector2_Add, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_Add)},
    {SIP_MLNAME_CAST(sipName_Find), meth_FbxLayerElementArrayTemplate_FbxVector2_Find, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_Find)},
    {SIP_MLNAME_CAST(sipName_FindAfter), meth_FbxLayerElementArrayTemplate_FbxVector2_FindAfter, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_FindAfter)},
    {SIP_MLNAME_CAST(sipName_FindBefore), meth_FbxLayerElementArrayTemplate_FbxVector2_FindBefore, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_FindBefore)},
    {SIP_MLNAME_CAST(sipName_GetAt), meth_FbxLayerElementArrayTemplate_FbxVector2_GetAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_GetAt)},
    {SIP_MLNAME_CAST(sipName_GetFirst), meth_FbxLayerElementArrayTemplate_FbxVector2_GetFirst, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_GetFirst)},
    {SIP_MLNAME_CAST(sipName_GetLast), meth_FbxLayerElementArrayTemplate_FbxVector2_GetLast, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_GetLast)},
    {SIP_MLNAME_CAST(sipName_InsertAt), meth_FbxLayerElementArrayTemplate_FbxVector2_InsertAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_InsertAt)},
    {SIP_MLNAME_CAST(sipName_RemoveAt), meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveAt)},
    {SIP_MLNAME_CAST(sipName_RemoveIt), meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveIt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveIt)},
    {SIP_MLNAME_CAST(sipName_RemoveLast), meth_FbxLayerElementArrayTemplate_FbxVector2_RemoveLast, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_RemoveLast)},
    {SIP_MLNAME_CAST(sipName_SetAt), meth_FbxLayerElementArrayTemplate_FbxVector2_SetAt, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_SetAt)},
    {SIP_MLNAME_CAST(sipName_SetLast), meth_FbxLayerElementArrayTemplate_FbxVector2_SetLast, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayerElementArrayTemplate_FbxVector2_SetLast)}
};

PyDoc_STRVAR(doc_FbxLayerElementArrayTemplate_FbxVector2, "\1FbxLayerElementArrayTemplate_FbxVector2(EFbxType)\n"
    "FbxLayerElementArrayTemplate_FbxVector2(FbxLayerElementArrayTemplate_FbxVector2)");


sipClassTypeDef sipTypeDef_fbx_FbxLayerElementArrayTemplate_FbxVector2 = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxLayerElementArrayTemplate_FbxVector2,
        {0},
        0
    },
    {
        sipNameNr_FbxLayerElementArrayTemplate_FbxVector2,
        {0, 0, 1},
        13, methods_FbxLayerElementArrayTemplate_FbxVector2,
        0, 0,
        0, 0,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    doc_FbxLayerElementArrayTemplate_FbxVector2,
    -1,
    -1,
    supers_FbxLayerElementArrayTemplate_FbxVector2,
    slots_FbxLayerElementArrayTemplate_FbxVector2,
    init_type_FbxLayerElementArrayTemplate_FbxVector2,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    dealloc_FbxLayerElementArrayTemplate_FbxVector2,
    0,
    0,
    0,
    release_FbxLayerElementArrayTemplate_FbxVector2,
    cast_FbxLayerElementArrayTemplate_FbxVector2,
    0,
    0,
    0,
    0,
    0,
    0
};
